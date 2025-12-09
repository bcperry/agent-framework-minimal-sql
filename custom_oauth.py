"""
Custom OAuth provider for Azure Government endpoints.
This module extends the Chainlit OAuth framework to support Azure Government Cloud
endpoints which differ from commercial Azure endpoints.
"""

import base64
import os
from typing import Dict, Tuple

import httpx
from fastapi import HTTPException

from chainlit.oauth_providers import OAuthProvider
from chainlit.secret import random_secret
from chainlit.user import User

ACCESS_TOKEN_MISSING = "Access token missing in the response"


class AzureGovOAuthProvider(OAuthProvider):
    """
    Azure Government OAuth Provider.

    Uses Azure Government Cloud endpoints (login.microsoftonline.us) instead of
    commercial Azure endpoints (login.microsoftonline.com).

    Environment variables required:
    - OAUTH_AZURE_GOV_AD_CLIENT_ID
    - OAUTH_AZURE_GOV_AD_CLIENT_SECRET
    - OAUTH_AZURE_GOV_AD_TENANT_ID
    """

    id = "azure-gov"
    env = [
        "OAUTH_AZURE_GOV_AD_CLIENT_ID",
        "OAUTH_AZURE_GOV_AD_CLIENT_SECRET",
        "OAUTH_AZURE_GOV_AD_TENANT_ID",
    ]

    # Azure Government Cloud endpoints
    AUTHORITY_URL = "https://login.microsoftonline.us"
    GRAPH_URL = "https://graph.microsoft.us"

    def __init__(self):
        self.client_id = os.environ.get("OAUTH_AZURE_GOV_AD_CLIENT_ID")
        self.client_secret = os.environ.get("OAUTH_AZURE_GOV_AD_CLIENT_SECRET")
        self.tenant_id = os.environ.get("OAUTH_AZURE_GOV_AD_TENANT_ID")

        # Build authorize URL
        self.authorize_url = (
            f"{self.AUTHORITY_URL}/{self.tenant_id}/oauth2/v2.0/authorize"
        )

        # Build token URL
        self.token_url = f"{self.AUTHORITY_URL}/{self.tenant_id}/oauth2/v2.0/token"

        self.authorize_params = {
            "response_type": "code",
            "scope": f"{self.GRAPH_URL}/User.Read offline_access",
            "response_mode": "query",
        }

        if prompt := self.get_prompt():
            self.authorize_params["prompt"] = prompt

    async def get_raw_token_response(self, code: str, url: str) -> dict:
        """Get the raw token response from Azure Government."""
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": url,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data=payload,
            )
            response.raise_for_status()
            return response.json()

    async def get_token(self, code: str, url: str) -> str:
        """Extract access token from the response."""
        json_response = await self.get_raw_token_response(code, url)
        token = json_response.get("access_token")
        refresh_token = json_response.get("refresh_token")

        if not token:
            raise HTTPException(status_code=400, detail=ACCESS_TOKEN_MISSING)

        # Store refresh token if available
        if refresh_token:
            self._refresh_token = refresh_token

        return token

    async def get_user_info(self, token: str) -> Tuple[Dict[str, str], User]:
        """Get user information from Microsoft Graph (Government)."""
        async with httpx.AsyncClient() as client:
            # Get user profile
            response = await client.get(
                f"{self.GRAPH_URL}/v1.0/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            azure_user = response.json()

            # Try to get user photo
            try:
                photo_response = await client.get(
                    f"{self.GRAPH_URL}/v1.0/me/photos/48x48/$value",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if photo_response.status_code == 200:
                    photo_data = await photo_response.aread()
                    base64_image = base64.b64encode(photo_data)
                    azure_user["image"] = (
                        f"data:{photo_response.headers.get('Content-Type', 'image/jpeg')};base64,"
                        f"{base64_image.decode('utf-8')}"
                    )
            except Exception:
                # Ignore errors getting the photo
                pass

            user = User(
                identifier=azure_user.get("userPrincipalName"),
                metadata={
                    "image": azure_user.get("image"),
                    "provider": "azure-gov",
                    "refresh_token": getattr(self, "_refresh_token", None),
                },
            )
            return (azure_user, user)


class AzureGovHybridOAuthProvider(OAuthProvider):
    """
    Azure Government Hybrid OAuth Provider (OAuth 2.0 + OpenID Connect).

    Supports hybrid flow with both authorization code and ID token for enhanced
    security. Uses Azure Government Cloud endpoints.

    Environment variables required:
    - OAUTH_AZURE_GOV_AD_HYBRID_CLIENT_ID
    - OAUTH_AZURE_GOV_AD_HYBRID_CLIENT_SECRET
    - OAUTH_AZURE_GOV_AD_HYBRID_TENANT_ID
    """

    id = "azure-gov-hybrid"
    env = [
        "OAUTH_AZURE_GOV_AD_HYBRID_CLIENT_ID",
        "OAUTH_AZURE_GOV_AD_HYBRID_CLIENT_SECRET",
        "OAUTH_AZURE_GOV_AD_HYBRID_TENANT_ID",
    ]

    # Azure Government Cloud endpoints
    AUTHORITY_URL = "https://login.microsoftonline.us"
    GRAPH_URL = "https://graph.microsoft.us"

    def __init__(self):
        self.client_id = os.environ.get("OAUTH_AZURE_GOV_AD_HYBRID_CLIENT_ID")
        self.client_secret = os.environ.get("OAUTH_AZURE_GOV_AD_HYBRID_CLIENT_SECRET")
        self.tenant_id = os.environ.get("OAUTH_AZURE_GOV_AD_HYBRID_TENANT_ID")

        # Build authorize URL
        self.authorize_url = (
            f"{self.AUTHORITY_URL}/{self.tenant_id}/oauth2/v2.0/authorize"
        )

        # Build token URL
        self.token_url = f"{self.AUTHORITY_URL}/{self.tenant_id}/oauth2/v2.0/token"

        nonce = random_secret(16)
        self.authorize_params = {
            "response_type": "code id_token",
            "scope": (
                f"{self.GRAPH_URL}/User.Read {self.GRAPH_URL}/openid offline_access"
            ),
            "response_mode": "form_post",
            "nonce": nonce,
        }

        if prompt := self.get_prompt():
            self.authorize_params["prompt"] = prompt

    async def get_raw_token_response(self, code: str, url: str) -> dict:
        """Get the raw token response from Azure Government."""
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": url,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data=payload,
            )
            response.raise_for_status()
            return response.json()

    async def get_token(self, code: str, url: str) -> str:
        """Extract access token from the response."""
        json_response = await self.get_raw_token_response(code, url)
        token = json_response.get("access_token")
        refresh_token = json_response.get("refresh_token")

        if not token:
            raise HTTPException(status_code=400, detail=ACCESS_TOKEN_MISSING)

        # Store refresh token if available
        if refresh_token:
            self._refresh_token = refresh_token

        return token

    async def get_user_info(self, token: str) -> Tuple[Dict[str, str], User]:
        """Get user information from Microsoft Graph (Government)."""
        async with httpx.AsyncClient() as client:
            # Get user profile
            response = await client.get(
                f"{self.GRAPH_URL}/v1.0/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            azure_user = response.json()

            # Try to get user photo
            try:
                photo_response = await client.get(
                    f"{self.GRAPH_URL}/v1.0/me/photos/48x48/$value",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if photo_response.status_code == 200:
                    photo_data = await photo_response.aread()
                    base64_image = base64.b64encode(photo_data)
                    azure_user["image"] = (
                        f"data:{photo_response.headers.get('Content-Type', 'image/jpeg')};base64,"
                        f"{base64_image.decode('utf-8')}"
                    )
            except Exception:
                # Ignore errors getting the photo
                pass

            user = User(
                identifier=azure_user.get("userPrincipalName"),
                metadata={
                    "image": azure_user.get("image"),
                    "provider": "azure-gov-hybrid",
                    "refresh_token": getattr(self, "_refresh_token", None),
                },
            )
            return (azure_user, user)
