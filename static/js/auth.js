/* =========================================================
   AUTH CLIENT-SIDE LOGIC
   Handles: login form, register form, password strength,
            real-time validation, show/hide password
   ========================================================= */

(function () {
    'use strict';

    /* ---- Helpers ---- */
    const $ = id => document.getElementById(id);
    const show = el => { if (el) el.style.display = 'flex'; };
    const hide = el => { if (el) el.style.display = 'none'; };

    function showError(msgEl, textEl, msg) {
        if (textEl) textEl.textContent = msg;
        if (msgEl)  show(msgEl);
    }

    function hideError(msgEl) {
        if (msgEl) hide(msgEl);
    }

    function setLoading(btnEl, textEl, loaderEl, loading) {
        if (!btnEl) return;
        btnEl.disabled = loading;
        if (loading) { hide(textEl); show(loaderEl); }
        else         { show(textEl); hide(loaderEl); }
    }

    /* ---- Toggle Password Visibility ---- */
    function bindToggle(btnId, inputId) {
        const btn   = $(btnId);
        const input = $(inputId);
        if (!btn || !input) return;
        btn.addEventListener('click', () => {
            const isText = input.type === 'text';
            input.type = isText ? 'password' : 'text';
            btn.textContent = isText ? '👁' : '🙈';
        });
    }

    bindToggle('toggle-pw-login',   'password');
    bindToggle('toggle-pw-reg',     'password');
    bindToggle('toggle-pw-confirm', 'confirm_password');

    /* ---- Password Strength ---- */
    const pwInput = $('password');
    const fill    = $('pw-strength-fill');
    const label   = $('pw-strength-label');

    function getStrength(pw) {
        if (!pw || pw.length < 4) return 'weak';
        const hasUpper   = /[A-Z]/.test(pw);
        const hasLower   = /[a-z]/.test(pw);
        const hasDigit   = /\d/.test(pw);
        const hasSpecial = /[^A-Za-z0-9]/.test(pw);
        const score = [pw.length >= 8, hasUpper, hasLower, hasDigit, hasSpecial]
                        .filter(Boolean).length;
        if (score <= 2) return 'weak';
        if (score <= 3) return 'medium';
        return 'strong';
    }

    if (pwInput && fill && label) {
        pwInput.addEventListener('input', () => {
            const strength = getStrength(pwInput.value);
            const labels   = { weak: 'Weak', medium: 'Fair', strong: 'Strong' };
            fill.className  = `pw-strength-fill ${strength}`;
            label.className = `pw-strength-label ${strength}`;
            label.textContent = pwInput.value ? labels[strength] : '';
        });
    }

    /* ---- Password Match Validation ---- */
    const confirmInput = $('confirm_password');
    const matchHint    = $('pw-match-hint');

    if (confirmInput && matchHint && pwInput) {
        confirmInput.addEventListener('input', checkMatch);
        pwInput.addEventListener('input', () => {
            if (confirmInput.value) checkMatch();
        });

        function checkMatch() {
            const match = pwInput.value === confirmInput.value;
            if (confirmInput.value === '') {
                matchHint.textContent = '';
                confirmInput.classList.remove('is-valid', 'is-invalid');
                return;
            }
            if (match) {
                matchHint.textContent = '✓ Passwords match';
                matchHint.className   = 'field-hint success';
                confirmInput.classList.add('is-valid');
                confirmInput.classList.remove('is-invalid');
            } else {
                matchHint.textContent = '✗ Passwords do not match';
                matchHint.className   = 'field-hint error';
                confirmInput.classList.add('is-invalid');
                confirmInput.classList.remove('is-valid');
            }
        }
    }

    /* ---- Username hint ---- */
    const usernameInput = $('username');
    const usernameHint  = $('username-hint');

    if (usernameInput && usernameHint) {
        usernameInput.addEventListener('input', () => {
            const val = usernameInput.value.trim();
            if (!val) {
                usernameHint.textContent = '';
                usernameInput.classList.remove('is-valid', 'is-invalid');
                return;
            }
            const valid = /^[a-zA-Z0-9_]{3,20}$/.test(val);
            if (valid) {
                usernameHint.textContent = '✓ Looks good';
                usernameHint.className   = 'field-hint success';
                usernameInput.classList.add('is-valid');
                usernameInput.classList.remove('is-invalid');
            } else {
                usernameHint.textContent = 'Use 3–20 letters, numbers or underscores';
                usernameHint.className   = 'field-hint error';
                usernameInput.classList.add('is-invalid');
                usernameInput.classList.remove('is-valid');
            }
        });
    }

    /* ---- Auto-dismiss flash messages ---- */
    const flashMsg = $('flash-msg');
    if (flashMsg) {
        setTimeout(() => {
            flashMsg.style.opacity = '0';
            flashMsg.style.transition = 'opacity 0.5s ease';
            setTimeout(() => flashMsg.remove(), 500);
        }, 4000);
    }

    /* ---- Login Form ---- */
    const loginForm = $('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const errorEl  = $('auth-error');
            const errorTxt = $('auth-error-text');
            hideError(errorEl);

            const username = $('username').value.trim();
            const password = $('password').value;

            if (!username || !password) {
                showError(errorEl, errorTxt, 'Please fill in all fields.');
                return;
            }

            const btnEl     = $('login-btn');
            const btnText   = $('login-btn-text');
            const btnLoader = $('login-btn-loader');
            setLoading(btnEl, btnText, btnLoader, true);

            try {
                const res = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username,
                        password,
                        remember: $('remember') ? $('remember').checked : false
                    })
                });

                const data = await res.json();

                if (res.ok && data.success) {
                    // Redirect to dashboard
                    window.location.href = data.redirect || '/';
                } else {
                    showError(errorEl, errorTxt, data.error || 'Invalid credentials. Please try again.');
                    setLoading(btnEl, btnText, btnLoader, false);
                }
            } catch (err) {
                showError(errorEl, errorTxt, 'Connection error. Please try again.');
                setLoading(btnEl, btnText, btnLoader, false);
            }
        });
    }

    /* ---- Register Form ---- */
    const registerForm = $('register-form');
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const errorEl   = $('auth-error');
            const errorTxt  = $('auth-error-text');
            const successEl = $('auth-success');
            const succTxt   = $('auth-success-text');

            hideError(errorEl);
            if (successEl) hide(successEl);

            const full_name        = $('full_name') ? $('full_name').value.trim() : '';
            const username         = $('username').value.trim();
            const email            = $('email') ? $('email').value.trim() : '';
            const password         = $('password').value;
            const confirm_password = $('confirm_password').value;

            // Client-side validation
            if (!full_name || !username || !password || !confirm_password) {
                showError(errorEl, errorTxt, 'Please fill in all required fields.');
                return;
            }

            if (!/^[a-zA-Z0-9_]{3,20}$/.test(username)) {
                showError(errorEl, errorTxt, 'Username must be 3–20 characters: letters, numbers, underscores only.');
                return;
            }

            if (password.length < 6) {
                showError(errorEl, errorTxt, 'Password must be at least 6 characters.');
                return;
            }

            if (password !== confirm_password) {
                showError(errorEl, errorTxt, 'Passwords do not match.');
                return;
            }

            const btnEl     = $('register-btn');
            const btnText   = $('register-btn-text');
            const btnLoader = $('register-btn-loader');
            setLoading(btnEl, btnText, btnLoader, true);

            try {
                const res = await fetch('/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ full_name, username, email, password, confirm_password })
                });

                const data = await res.json();

                if (res.ok && data.success) {
                    if (successEl && succTxt) {
                        succTxt.textContent = data.message || 'Account created! Redirecting…';
                        show(successEl);
                    }
                    setTimeout(() => {
                        window.location.href = data.redirect || '/login';
                    }, 1200);
                } else {
                    showError(errorEl, errorTxt, data.error || 'Registration failed. Please try again.');
                    setLoading(btnEl, btnText, btnLoader, false);
                }
            } catch (err) {
                showError(errorEl, errorTxt, 'Connection error. Please try again.');
                setLoading(btnEl, btnText, btnLoader, false);
            }
        });
    }

})();
