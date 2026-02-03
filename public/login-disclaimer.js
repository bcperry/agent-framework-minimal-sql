// Inject full-screen disclaimer overlay on login page ONLY
(function() {
    // Use sessionStorage to persist acknowledgment across page loads
    const STORAGE_KEY = 'login_disclaimer_acknowledged';

    function isLoginPage() {
        // Be very strict - only show on exactly /login
        return window.location.pathname === '/login' || window.location.pathname === '/login/';
    }

    function isAcknowledged() {
        return sessionStorage.getItem(STORAGE_KEY) === 'true';
    }

    function acknowledgeDisclaimer() {
        sessionStorage.setItem(STORAGE_KEY, 'true');
        const overlay = document.querySelector('.login-disclaimer-overlay');
        if (overlay) {
            overlay.remove();
        }
    }

    // Expose to global scope for button onclick
    window.acknowledgeLoginDisclaimer = acknowledgeDisclaimer;

    // Simple markdown to HTML converter
    function markdownToHtml(md) {
        return md
            // Remove the outer div tags if present
            .replace(/<div class="disclaimer">/g, '')
            .replace(/<\/div>/g, '')
            // Headers
            .replace(/^## (.+)$/gm, '<h2>$1</h2>')
            .replace(/^### (.+)$/gm, '<h3>$1</h3>')
            // Bold
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            // List items
            .replace(/^- (.+)$/gm, '<li>$1</li>')
            // Wrap consecutive li in ul
            .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
            // Paragraphs (lines that aren't already wrapped)
            .replace(/^(?!<[hul]|<li)(.+)$/gm, '<p>$1</p>')
            // Clean up empty paragraphs
            .replace(/<p>\s*<\/p>/g, '')
            .trim();
    }

    async function loadDisclaimerContent() {
        try {
            const response = await fetch('/public/disclaimer.txt');
            if (!response.ok) {
                throw new Error('Failed to load disclaimer');
            }
            const text = await response.text();
            return markdownToHtml(text);
        } catch (error) {
            console.error('Error loading disclaimer:', error);
            // Fallback content
            return `
                <h2>⚠️ WARNING ⚠️</h2>
                <p><strong>You are accessing a proprietary information system.</strong></p>
                <p>By continuing, you consent to monitoring and agree to acceptable use policies.</p>
            `;
        }
    }

    async function addDisclaimer() {
        // Don't show if already acknowledged
        if (isAcknowledged()) {
            return;
        }

        // ONLY run on login page - strict check
        if (!isLoginPage()) {
            return;
        }

        // Check if disclaimer already exists
        if (document.querySelector('.login-disclaimer-overlay')) {
            return;
        }

        // Load disclaimer content from file
        const disclaimerContent = await loadDisclaimerContent();

        // Create full-screen overlay
        const overlay = document.createElement('div');
        overlay.className = 'login-disclaimer-overlay';
        overlay.innerHTML = `
            <div class="disclaimer-container">
                <div class="disclaimer-content">
                    ${disclaimerContent}
                    
                    <button class="acknowledge-btn" onclick="window.acknowledgeLoginDisclaimer()">
                        I UNDERSTAND AND AGREE
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);
    }

    // Only run on initial page load if we're on login page
    if (isLoginPage() && !isAcknowledged()) {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', addDisclaimer);
        } else {
            addDisclaimer();
        }
    }
})();
