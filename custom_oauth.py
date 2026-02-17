"""
Custom OAuth provider for Azure Government endpoints.
This module extends the Chainlit OAuth framework to support Azure Government Cloud
endpoints which differ from commercial Azure endpoints.

Supports certificate-based authentication (client_assertion) for confidential clients
without client secrets.
"""

import base64
import hashlib
import logging
import os
import time
import uuid
from typing import Dict, Optional, Tuple

import httpx
from fastapi import HTTPException

from chainlit.oauth_providers import OAuthProvider
from chainlit.secret import random_secret
from chainlit.user import User

ACCESS_TOKEN_MISSING = "Access token missing in the response"

logger = logging.getLogger(__name__)


def create_client_assertion(client_id: str, tenant_id: str, private_key_path: str, cert_thumbprint: str, authority_url: str) -> str:
    """Create a client assertion JWT for certificate-based authentication.
    
    Args:
        client_id: The Azure AD application client ID
        tenant_id: The Azure AD tenant ID
        private_key_path: Path to the private key PEM file
        cert_thumbprint: SHA-1 thumbprint of the certificate (hex, uppercase)
        authority_url: The Azure authority URL (e.g., https://login.microsoftonline.us)
    
    Returns:
        A signed JWT client assertion
    """
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography import x509
    
    # Load the private key
    with open(private_key_path, 'rb') as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)
    
    # Get the certificate thumbprint from the cert file if not provided
    if not cert_thumbprint:
        cert_path = private_key_path.replace('_private.pem', '_cert.pem')
        with open(cert_path, 'rb') as f:
            cert = x509.load_pem_x509_certificate(f.read())
            cert_thumbprint = cert.fingerprint(cert.signature_hash_algorithm).hex().upper()
    
    # Build the JWT header with x5t (certificate thumbprint)
    # x5t is base64url-encoded SHA-1 thumbprint
    thumbprint_bytes = bytes.fromhex(cert_thumbprint)
    x5t = base64.urlsafe_b64encode(thumbprint_bytes).decode().rstrip('=')
    
    headers = {
        "alg": "RS256",
        "typ": "JWT",
        "x5t": x5t,
    }
    
    # Build the JWT payload
    now = int(time.time())
    token_url = f"{authority_url}/{tenant_id}/oauth2/v2.0/token"
    
    payload = {
        "aud": token_url,
        "exp": now + 600,  # 10 minutes
        "iss": client_id,
        "jti": str(uuid.uuid4()),
        "nbf": now,
        "sub": client_id,
        "iat": now,
    }
    
    # Sign and return the JWT
    return jwt.encode(payload, private_key, algorithm="RS256", headers=headers)


class AzureGovOAuthProvider(OAuthProvider):
    """
    Azure Government OAuth Provider with certificate-based authentication.

    Uses Azure Government Cloud endpoints (login.microsoftonline.us) instead of
    commercial Azure endpoints (login.microsoftonline.com).

    Supports certificate-based authentication (client_assertion) for confidential
    clients without client secrets.

    Environment variables required:
    - OAUTH_AZURE_GOV_AD_CLIENT_ID
    - OAUTH_AZURE_GOV_AD_TENANT_ID
    
    For certificate authentication:
    - OAUTH_AZURE_GOV_AD_CERT_PATH (path to private key PEM file)
    - OAUTH_AZURE_GOV_AD_CERT_THUMBPRINT (SHA-1 thumbprint of the certificate)
    
    Optional (for confidential clients with secrets):
    - OAUTH_AZURE_GOV_AD_CLIENT_SECRET
    """

    id = "azure-gov"
    env = [
        "OAUTH_AZURE_GOV_AD_CLIENT_ID",
        "OAUTH_AZURE_GOV_AD_TENANT_ID",
    ]

    # Azure Government Cloud endpoints
    AUTHORITY_URL = "https://login.microsoftonline.us"
    GRAPH_URL = "https://graph.microsoft.us"

    def __init__(self):
        self.client_id = os.environ.get("OAUTH_AZURE_GOV_AD_CLIENT_ID")
        self.client_secret = os.environ.get("OAUTH_AZURE_GOV_AD_CLIENT_SECRET")
        self.tenant_id = os.environ.get("OAUTH_AZURE_GOV_AD_TENANT_ID")
        self.cert_path = os.environ.get("OAUTH_AZURE_GOV_AD_CERT_PATH")
        self.cert_thumbprint = os.environ.get("OAUTH_AZURE_GOV_AD_CERT_THUMBPRINT")
        
        # Determine authentication method
        self.use_cert = bool(self.cert_path and os.path.exists(self.cert_path))
        self.use_secret = bool(self.client_secret)

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
        
        auth_method = "certificate" if self.use_cert else ("secret" if self.use_secret else "none")
        logger.info(f"[azure-gov] Initialized with auth_method={auth_method}, client_id={self.client_id[:8] if self.client_id else 'None'}...")

    async def get_raw_token_response(self, code: str, url: str) -> dict:
        """Get the raw token response from Azure Government."""
        payload = {
            "client_id": self.client_id,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": url,
        }
        
        # Add authentication - prefer certificate, then secret
        if self.use_cert:
            client_assertion = create_client_assertion(
                client_id=self.client_id,
                tenant_id=self.tenant_id,
                private_key_path=self.cert_path,
                cert_thumbprint=self.cert_thumbprint,
                authority_url=self.AUTHORITY_URL
            )
            payload["client_assertion"] = client_assertion
            payload["client_assertion_type"] = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            logger.info(f"[azure-gov] Using certificate (client_assertion) for token exchange")
        elif self.use_secret:
            payload["client_secret"] = self.client_secret
            logger.info(f"[azure-gov] Using client_secret for token exchange")
        else:
            logger.error(f"[azure-gov] No authentication method configured!")
        
        logger.info(f"[azure-gov] Token URL: {self.token_url}")
        logger.info(f"[azure-gov] Redirect URI: {url}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data=payload,
            )
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"[azure-gov] Token exchange failed ({response.status_code}): {error_text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Azure token exchange failed: {error_text}"
                )
            return response.json()

    async def get_token(self, code: str, url: str) -> str:
        """Extract access token from the response."""
        json_response = await self.get_raw_token_response(code, url)
        token = json_response.get("access_token")

        if not token:
            raise HTTPException(status_code=400, detail=ACCESS_TOKEN_MISSING)

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
                },
            )
            return (azure_user, user)


class AzureGovHybridOAuthProvider(OAuthProvider):
    """
    Azure Government Hybrid OAuth Provider (OAuth 2.0 + OpenID Connect) with certificate support.

    Supports hybrid flow with both authorization code and ID token for enhanced
    security. Uses Azure Government Cloud endpoints.

    Supports certificate-based authentication (client_assertion) for confidential
    clients without client secrets.

    Environment variables required:
    - OAUTH_AZURE_GOV_AD_HYBRID_CLIENT_ID
    - OAUTH_AZURE_GOV_AD_HYBRID_TENANT_ID
    
    For certificate authentication:
    - OAUTH_AZURE_GOV_AD_HYBRID_CERT_PATH (path to private key PEM file)
    - OAUTH_AZURE_GOV_AD_HYBRID_CERT_THUMBPRINT (SHA-1 thumbprint of the certificate)
    
    Optional (for confidential clients with secrets):
    - OAUTH_AZURE_GOV_AD_HYBRID_CLIENT_SECRET
    """

    id = "azure-gov-hybrid"
    env = [
        "OAUTH_AZURE_GOV_AD_HYBRID_CLIENT_ID",
        "OAUTH_AZURE_GOV_AD_HYBRID_TENANT_ID",
    ]

    # Azure Government Cloud endpoints
    AUTHORITY_URL = "https://login.microsoftonline.us"
    GRAPH_URL = "https://graph.microsoft.us"

    def __init__(self):
        self.client_id = os.environ.get("OAUTH_AZURE_GOV_AD_HYBRID_CLIENT_ID")
        self.client_secret = os.environ.get("OAUTH_AZURE_GOV_AD_HYBRID_CLIENT_SECRET")
        self.tenant_id = os.environ.get("OAUTH_AZURE_GOV_AD_HYBRID_TENANT_ID")
        self.cert_path = os.environ.get("OAUTH_AZURE_GOV_AD_HYBRID_CERT_PATH")
        self.cert_thumbprint = os.environ.get("OAUTH_AZURE_GOV_AD_HYBRID_CERT_THUMBPRINT")
        
        # Determine authentication method
        self.use_cert = bool(self.cert_path and os.path.exists(self.cert_path))
        self.use_secret = bool(self.client_secret)

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
        
        auth_method = "certificate" if self.use_cert else ("secret" if self.use_secret else "none")
        logger.info(f"[azure-gov-hybrid] Initialized with auth_method={auth_method}")

    async def get_raw_token_response(self, code: str, url: str) -> dict:
        """Get the raw token response from Azure Government."""
        payload = {
            "client_id": self.client_id,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": url,
        }
        
        # Add authentication - prefer certificate, then secret
        if self.use_cert:
            client_assertion = create_client_assertion(
                client_id=self.client_id,
                tenant_id=self.tenant_id,
                private_key_path=self.cert_path,
                cert_thumbprint=self.cert_thumbprint,
                authority_url=self.AUTHORITY_URL
            )
            payload["client_assertion"] = client_assertion
            payload["client_assertion_type"] = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            logger.info(f"[azure-gov-hybrid] Using certificate (client_assertion) for token exchange")
        elif self.use_secret:
            payload["client_secret"] = self.client_secret
            logger.info(f"[azure-gov-hybrid] Using client_secret for token exchange")
        else:
            logger.error(f"[azure-gov-hybrid] No authentication method configured!")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data=payload,
            )
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"[azure-gov-hybrid] Token exchange failed ({response.status_code}): {error_text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Azure token exchange failed: {error_text}"
                )
            return response.json()

    async def get_token(self, code: str, url: str) -> str:
        """Extract access token from the response."""
        json_response = await self.get_raw_token_response(code, url)
        token = json_response.get("access_token")

        if not token:
            raise HTTPException(status_code=400, detail=ACCESS_TOKEN_MISSING)

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
                },
            )
            return (azure_user, user)
