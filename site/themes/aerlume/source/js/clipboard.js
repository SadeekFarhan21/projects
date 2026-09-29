// 代码块增强：复制 / 查看源码（无 jQuery、无 Font Awesome 依赖，纯内联 SVG 图标）
!function () {
    // 内联 SVG 图标（与导航栏风格一致，stroke-based）
    var ICONS = {
        copy: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>',
        check: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
        warn: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
    };

    // The button's own label change is not announced by screen readers, so the
    // result is also spoken through one visually hidden live region.
    var liveEl = null;
    function announce(msg) {
        if (!liveEl) {
            liveEl = document.createElement('div');
            liveEl.setAttribute('role', 'status');
            liveEl.setAttribute('aria-live', 'polite');
            liveEl.style.cssText = 'position:absolute;width:1px;height:1px;margin:-1px;padding:0;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;border:0';
            document.body.appendChild(liveEl);
        }
        liveEl.textContent = '';
        setTimeout(function () { liveEl.textContent = msg; }, 50);
    }

    // 通用复制：优先 navigator.clipboard，回退 execCommand
    function copyText(text, btn, doneIcon, doneLabel) {
        var onDone = function () {
            announce('Code copied to clipboard');
            setBtnState(btn, doneIcon || ICONS.check, doneLabel || 'Copied', true);
            setTimeout(function () {
                setBtnState(btn, ICONS.copy, 'Copy', false);
            }, 2000);
        };
        var onFail = function () {
            announce('Copy failed');
            setBtnState(btn, ICONS.warn, 'Failed', true);
            setTimeout(function () {
                setBtnState(btn, ICONS.copy, 'Copy', false);
            }, 2000);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(onDone, function () { fallbackCopy(text, onDone, onFail); });
        } else {
            fallbackCopy(text, onDone, onFail);
        }
    }

    function fallbackCopy(text, onDone, onFail) {
        try {
            var ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.left = '-9999px';
            ta.style.top = '0';
            document.body.appendChild(ta);
            ta.focus();
            ta.select();
            var ok = document.execCommand('copy');
            document.body.removeChild(ta);
            ok ? onDone() : onFail();
        } catch (e) {
            onFail();
        }
    }

    function setBtnState(btn, iconSvg, label, copied) {
        var icon = btn.querySelector('svg');
        var span = btn.querySelector('span');
        if (icon) icon.outerHTML = iconSvg;
        if (span) span.textContent = label;
        if (copied) btn.classList.add('is-copied');
        else btn.classList.remove('is-copied');
    }

    function buildActionsHtml(lang) {
        var shown = lang === 'plaintext' ? 'text' : lang;
        var langLabel = shown ? '<span class="code-lang">' + escapeHtml(shown) + '</span>' : '';
        return '<div class="code-actions">'
            + langLabel
            + '<button class="btn-copy" type="button" title="Copy code">' + ICONS.copy + '<span>Copy</span></button>'
            + '</div>';
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"]/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
        });
    }

    // 从 Hexo highlight.js 输出中识别语言：figure.highlight 的类名形如 "highlight python"
    function detectLang(block) {
        var m = block.className && block.className.match(/(?:^|\s)highlight(?:\s+([\w-]+))?/);
        if (m && m[1]) return m[1];
        var figcaption = block.querySelector('figcaption');
        if (figcaption) {
            var t = figcaption.textContent.trim();
            if (t) return t;
        }
        var codeEl = block.querySelector('code');
        if (codeEl) {
            var cm = codeEl.className && codeEl.className.match(/language-([\w-]+)/);
            if (cm) return cm[1];
        }
        return '';
    }

    function getCodeText(block) {
        var codeEl = block.querySelector('.code') || block.querySelector('pre code') || block.querySelector('code');
        return codeEl ? codeEl.innerText : '';
    }

    function initCopyCode() {
        // Code blocks are shown in full in the post; there is no pop-up view,
        // only a Copy button.
        // 为每个代码块注入操作按钮
        document.querySelectorAll('.highlight').forEach(function (block) {
            if (block.querySelector('.code-actions')) return; // 避免重复注入
            var lang = detectLang(block);
            block.insertAdjacentHTML('afterbegin', buildActionsHtml(lang));

            block.querySelector('.btn-copy').addEventListener('click', function () {
                copyText(getCodeText(block), block.querySelector('.btn-copy'));
            });
        });
    }

    // 兼容历史：若页面已加载 ClipboardJS，仍可工作；否则用上面的原生实现
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCopyCode);
    } else {
        initCopyCode();
    }

    // 加密文章解密后，为新出现的代码块注入操作按钮（已有防重复注入守卫）
    window.addEventListener('hexo-blog-decrypt', function () {
        initCopyCode();
    });
}();
