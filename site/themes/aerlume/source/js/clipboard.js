// 代码块增强：复制 / 查看源码（无 jQuery、无 Font Awesome 依赖，纯内联 SVG 图标）
!function () {
    // 内联 SVG 图标（与导航栏风格一致，stroke-based）
    var ICONS = {
        copy: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>',
        check: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
        expand: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>',
        close: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>',
        warn: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
    };

    // 通用复制：优先 navigator.clipboard，回退 execCommand
    function copyText(text, btn, doneIcon, doneLabel) {
        var onDone = function () {
            setBtnState(btn, doneIcon || ICONS.check, doneLabel || 'Copied', true);
            setTimeout(function () {
                setBtnState(btn, ICONS.copy, 'Copy', false);
            }, 2000);
        };
        var onFail = function () {
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
        var langLabel = lang ? '<span class="code-lang">' + escapeHtml(lang) + '</span>' : '';
        return '<div class="code-actions">'
            + langLabel
            + '<button class="btn-copy" type="button" title="Copy code">' + ICONS.copy + '<span>Copy</span></button>'
            + '<button class="btn-view-raw" type="button" title="View source">' + ICONS.expand + '<span>View</span></button>'
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
        // 创建查看源码模态弹窗（全局唯一）
        if (!document.getElementById('code-modal')) {
            document.body.insertAdjacentHTML('beforeend',
                '<div class="code-modal" id="code-modal">'
                + '<div class="code-modal-overlay"></div>'
                + '<div class="code-modal-content">'
                + '  <div class="code-modal-header">'
                + '    <span class="code-modal-title">Code</span>'
                + '    <button class="code-modal-copy" type="button">' + ICONS.copy + '<span>Copy</span></button>'
                + '    <button class="code-modal-close" type="button">' + ICONS.close + '</button>'
                + '  </div>'
                + '  <div class="code-modal-body"><pre></pre></div>'
                + '</div></div>');
        }

        var modal = document.getElementById('code-modal');
        var modalOverlay = modal.querySelector('.code-modal-overlay');
        var modalClose = modal.querySelector('.code-modal-close');
        var modalCopy = modal.querySelector('.code-modal-copy');
        var modalBody = modal.querySelector('.code-modal-body pre');
        var modalTitle = modal.querySelector('.code-modal-title');

        function closeModal() {
            modal.classList.remove('is-visible');
            document.body.style.overflow = '';
        }
        function openModal(code, lang) {
            modalTitle.textContent = lang ? 'Code — ' + lang : 'Code';
            modalBody.textContent = code;
            modal.classList.add('is-visible');
            document.body.style.overflow = 'hidden';
        }

        modalOverlay.addEventListener('click', closeModal);
        modalClose.addEventListener('click', closeModal);
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && modal.classList.contains('is-visible')) closeModal();
        });
        modalCopy.addEventListener('click', function () {
            copyText(modalBody.textContent, modalCopy, ICONS.check, 'Copied');
        });

        // 为每个代码块注入操作按钮
        document.querySelectorAll('.highlight').forEach(function (block) {
            if (block.querySelector('.code-actions')) return; // 避免重复注入
            var lang = detectLang(block);
            block.insertAdjacentHTML('afterbegin', buildActionsHtml(lang));

            block.querySelector('.btn-view-raw').addEventListener('click', function () {
                openModal(getCodeText(block), lang);
            });
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
