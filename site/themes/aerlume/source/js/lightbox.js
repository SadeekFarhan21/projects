// 轻量图片灯箱 / 图库查看器（无 jQuery、无外部依赖）
// 功能：上一张/下一张、缩放、拖拽平移、键盘、标题、计数、加载态、触屏滑动
!function () {
    var ICONS = {
        close: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>',
        prev: '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>',
        next: '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>',
        zoomIn: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>',
        zoomOut: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>'
    };

    // The dialog fades in with a visibility transition, so it can still be
    // "hidden" (and unfocusable) for the first frame; retry for a few frames.
    function focusWhenShown(el, box) {
        var tries = 0;
        (function attempt() {
            if (!box.classList.contains('is-visible')) return;
            el.focus();
            if (document.activeElement !== el && tries++ < 30) requestAnimationFrame(attempt);
        })();
    }

    var overlay = null;
    var imgEl = null;
    var counterEl = null;
    var captionEl = null;
    var spinnerEl = null;
    var prevBtn = null;
    var nextBtn = null;
    var zoomBtn = null;
    var hintEl = null;

    var images = [];        // [{src, alt}]
    var current = 0;

    // 缩放 / 平移状态
    var scale = 1;
    var tx = 0;
    var ty = 0;
    var MAX_SCALE = 3;

    // 拖拽状态
    var dragging = false;
    var dragStartX = 0;
    var dragStartY = 0;
    var dragStartTx = 0;
    var dragStartTy = 0;
    var movedDuringDrag = false;

    // 触屏滑动状态
    var touchStartX = 0;
    var touchStartY = 0;
    var touchStartTime = 0;

    function buildOverlay() {
        overlay = document.createElement('div');
        overlay.className = 'lb-overlay';
        overlay.setAttribute('role', 'dialog');
        overlay.setAttribute('aria-modal', 'true');
        overlay.setAttribute('aria-label', 'Image viewer');

        spinnerEl = document.createElement('div');
        spinnerEl.className = 'lb-spinner';
        overlay.appendChild(spinnerEl);

        var topbar = document.createElement('div');
        topbar.className = 'lb-topbar';
        counterEl = document.createElement('span');
        counterEl.className = 'lb-counter';
        captionEl = document.createElement('span');
        captionEl.className = 'lb-caption';
        var closeBtn = document.createElement('button');
        closeBtn.className = 'lb-close';
        closeBtn.type = 'button';
        closeBtn.setAttribute('aria-label', 'Close');
        closeBtn.innerHTML = ICONS.close;
        closeBtn.addEventListener('click', function (e) { e.stopPropagation(); close(); });
        topbar.appendChild(counterEl);
        topbar.appendChild(captionEl);
        topbar.appendChild(closeBtn);
        overlay.appendChild(topbar);

        prevBtn = document.createElement('button');
        prevBtn.className = 'lb-nav lb-prev';
        prevBtn.type = 'button';
        prevBtn.setAttribute('aria-label', 'Previous image');
        prevBtn.innerHTML = ICONS.prev;
        prevBtn.addEventListener('click', function (e) { e.stopPropagation(); prev(); });
        overlay.appendChild(prevBtn);

        nextBtn = document.createElement('button');
        nextBtn.className = 'lb-nav lb-next';
        nextBtn.type = 'button';
        nextBtn.setAttribute('aria-label', 'Next image');
        nextBtn.innerHTML = ICONS.next;
        nextBtn.addEventListener('click', function (e) { e.stopPropagation(); next(); });
        overlay.appendChild(nextBtn);

        imgEl = document.createElement('img');
        imgEl.className = 'lb-img';
        overlay.appendChild(imgEl);

        zoomBtn = document.createElement('button');
        zoomBtn.className = 'lb-zoom';
        zoomBtn.type = 'button';
        zoomBtn.setAttribute('aria-label', 'Zoom in');
        zoomBtn.innerHTML = ICONS.zoomIn;
        zoomBtn.addEventListener('click', function (e) { e.stopPropagation(); toggleZoom(); });
        overlay.appendChild(zoomBtn);

        hintEl = document.createElement('div');
        hintEl.className = 'lb-hint';
        hintEl.textContent = 'Click or + − to zoom · ← → to switch · Esc to close';
        overlay.appendChild(hintEl);

        // 点击空白处（非图片、非按钮）关闭
        overlay.addEventListener('click', function (e) {
            if (e.target === overlay || e.target === spinnerEl) close();
        });

        document.body.appendChild(overlay);

        // 图片交互
        imgEl.addEventListener('click', onImageClick);
        imgEl.addEventListener('dblclick', onImageDblClick);
        imgEl.addEventListener('load', onImageLoad);
        // 加载失败时不要让加载动画一直转
        imgEl.addEventListener('error', function () { spinnerEl.style.display = 'none'; });

        // 鼠标拖拽平移（缩放态）
        imgEl.addEventListener('mousedown', onDragStart);
        window.addEventListener('mousemove', onDragMove);
        window.addEventListener('mouseup', onDragEnd);

        // 触屏
        imgEl.addEventListener('touchstart', onTouchStart, { passive: false });
        imgEl.addEventListener('touchmove', onTouchMove, { passive: false });
        imgEl.addEventListener('touchend', onTouchEnd);

        // 鼠标滚轮缩放
        overlay.addEventListener('wheel', onWheel, { passive: false });
    }

    function onImageLoad() {
        spinnerEl.style.display = 'none';
        imgEl.classList.add('is-loaded');
    }

    function onImageClick(e) {
        e.stopPropagation();
        if (movedDuringDrag) { movedDuringDrag = false; return; }
        toggleZoom();
    }

    function onImageDblClick(e) {
        e.stopPropagation();
        e.preventDefault();
    }

    function toggleZoom() {
        if (scale > 1) {
            setZoom(1, 0, 0);
        } else {
            setZoom(2, 0, 0);
        }
    }

    function setZoom(s, x, y) {
        scale = s;
        // 回到原始大小时同时复位平移，否则图片停在偏移位置且无法再拖回
        tx = scale > 1 ? x : 0;
        ty = scale > 1 ? y : 0;
        applyTransform();
        if (scale > 1) clampPan();
        zoomBtn.innerHTML = scale > 1 ? ICONS.zoomOut : ICONS.zoomIn;
        zoomBtn.setAttribute('aria-label', scale > 1 ? 'Zoom out' : 'Zoom in');
        if (scale > 1) {
            imgEl.classList.add('is-zoomed');
            overlay.classList.add('is-zoomed');
        } else {
            imgEl.classList.remove('is-zoomed');
            overlay.classList.remove('is-zoomed');
        }
    }

    function applyTransform() {
        imgEl.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + scale + ')';
    }

    function clampPan() {
        // 限制平移范围，避免图片被拖出视口
        var rect = imgEl.getBoundingClientRect();
        var vw = window.innerWidth;
        var vh = window.innerHeight;
        var maxX = Math.max(0, (rect.width - vw) / 2);
        var maxY = Math.max(0, (rect.height - vh) / 2);
        if (tx > maxX) tx = maxX;
        if (tx < -maxX) tx = -maxX;
        if (ty > maxY) ty = maxY;
        if (ty < -maxY) ty = -maxY;
        applyTransform();
    }

    function onDragStart(e) {
        if (scale <= 1) return;
        e.preventDefault();
        dragging = true;
        movedDuringDrag = false;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        dragStartTx = tx;
        dragStartTy = ty;
        imgEl.style.cursor = 'grabbing';
    }

    function onDragMove(e) {
        if (!dragging) return;
        var dx = e.clientX - dragStartX;
        var dy = e.clientY - dragStartY;
        if (Math.abs(dx) > 3 || Math.abs(dy) > 3) movedDuringDrag = true;
        tx = dragStartTx + dx;
        ty = dragStartTy + dy;
        applyTransform();
    }

    function onDragEnd() {
        if (!dragging) return;
        dragging = false;
        imgEl.style.cursor = scale > 1 ? 'grab' : 'zoom-in';
        clampPan();
    }

    function onWheel(e) {
        if (!overlay || !overlay.classList.contains('is-visible')) return;
        e.preventDefault();
        var delta = e.deltaY < 0 ? 0.2 : -0.2;
        var ns = Math.max(1, Math.min(MAX_SCALE, scale + delta));
        setZoom(ns, tx, ty);
    }

    function onTouchStart(e) {
        if (e.touches.length === 1) {
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
            touchStartTime = Date.now();
            if (scale > 1) {
                dragging = true;
                movedDuringDrag = false;
                dragStartX = touchStartX;
                dragStartY = touchStartY;
                dragStartTx = tx;
                dragStartTy = ty;
            }
        }
    }

    function onTouchMove(e) {
        if (e.touches.length !== 1) return;
        var dx = e.touches[0].clientX - touchStartX;
        var dy = e.touches[0].clientY - touchStartY;
        if (scale > 1 && dragging) {
            e.preventDefault();
            if (Math.abs(dx) > 3 || Math.abs(dy) > 3) movedDuringDrag = true;
            tx = dragStartTx + dx;
            ty = dragStartTy + dy;
            applyTransform();
        }
    }

    function onTouchEnd(e) {
        dragging = false;
        var dx = (e.changedTouches[0] || {}).clientX - touchStartX;
        var dy = (e.changedTouches[0] || {}).clientY - touchStartY;
        var dt = Date.now() - touchStartTime;
        // 缩放态：拖拽结束后夹紧；轻触则切换缩放
        // 已在此处理的轻触/滑动要阻止随后的合成 click，否则 onImageClick 会再切换一次缩放
        if (scale > 1) {
            clampPan();
            if (!movedDuringDrag && dt < 250 && Math.abs(dx) < 10 && Math.abs(dy) < 10) {
                if (e.cancelable) e.preventDefault();
                toggleZoom();
            }
            return;
        }
        // 非缩放态：水平滑动切换图片
        if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy)) {
            if (e.cancelable) e.preventDefault();
            if (dx < 0) next(); else prev();
        } else if (!movedDuringDrag && dt < 250 && Math.abs(dx) < 10 && Math.abs(dy) < 10) {
            // 轻触切换缩放
            if (e.cancelable) e.preventDefault();
            toggleZoom();
        }
    }

    function goTo(index) {
        current = (index + images.length) % images.length;
        var item = images[current];
        spinnerEl.style.display = 'block';
        imgEl.classList.remove('is-loaded');
        imgEl.src = item.src;
        imgEl.alt = item.alt || '';
        counterEl.textContent = (current + 1) + ' / ' + images.length;
        captionEl.textContent = item.alt || '';
        captionEl.style.display = item.alt ? '' : 'none';
        // 切换图片时重置缩放
        setZoom(1, 0, 0);
        // 只有一张时隐藏导航
        var single = images.length <= 1;
        prevBtn.style.display = single ? 'none' : '';
        nextBtn.style.display = single ? 'none' : '';
    }

    function prev() { goTo(current - 1); }
    function next() { goTo(current + 1); }

    var returnFocus = null;

    function open(index) {
        if (!overlay) buildOverlay();
        returnFocus = document.activeElement;
        goTo(index);
        overlay.classList.add('is-visible');
        document.body.style.overflow = 'hidden';
        focusWhenShown(overlay.querySelector('.lb-close'), overlay);
    }

    function close() {
        if (!overlay || !overlay.classList.contains('is-visible')) return;
        overlay.classList.remove('is-visible');
        document.body.style.overflow = '';
        setZoom(1, 0, 0);
        if (returnFocus && document.contains(returnFocus)) returnFocus.focus();
        returnFocus = null;
    }

    // 焦点限制在灯箱内的可见按钮之间
    function trapTab(e) {
        var buttons = Array.from(overlay.querySelectorAll('button')).filter(function (b) {
            return b.getClientRects().length > 0;
        });
        if (!buttons.length) return;
        var i = buttons.indexOf(document.activeElement);
        e.preventDefault();
        if (e.shiftKey) i = i <= 0 ? buttons.length - 1 : i - 1;
        else i = i < 0 || i === buttons.length - 1 ? 0 : i + 1;
        buttons[i].focus();
    }

    // 键盘
    document.addEventListener('keydown', function (e) {
        if (!overlay || !overlay.classList.contains('is-visible')) return;
        switch (e.key) {
            case 'Tab': trapTab(e); break;
            case 'Escape': close(); break;
            case 'ArrowLeft': prev(); break;
            case 'ArrowRight': next(); break;
            case '+': case '=':
                setZoom(Math.min(MAX_SCALE, scale + 0.5), tx, ty); break;
            case '-': case '_':
                setZoom(Math.max(1, scale - 0.5), tx, ty); break;
            case '0':
                setZoom(1, 0, 0); break;
        }
    });

    // 窗口尺寸变化时重新夹紧
    window.addEventListener('resize', function () {
        if (overlay && overlay.classList.contains('is-visible') && scale > 1) clampPan();
    });

    // 收集文章内的图片，建立图库；点击打开
    var imgs = [];
    var clickBound = false;

    function collectImages() {
        var container = document.querySelector('.post-content') || document.querySelector('.index-middle');
        if (!container) return;

        imgs = Array.from(container.querySelectorAll('img')).filter(function (img) {
            // 排除赞赏二维码等
            if (img.closest('.donate-container')) return false;
            // 排除极小图标
            if (img.width && img.width < 24 && img.height && img.height < 24) return false;
            return true;
        });

        imgs.forEach(function (img, i) {
            img.style.cursor = 'zoom-in';
            img.dataset.lbIndex = i;
            // 键盘可达：不在链接里的图片可用 Tab 聚焦、Enter/空格打开
            if (!img.closest('a, button') && !img.hasAttribute('tabindex')) {
                img.tabIndex = 0;
                img.setAttribute('role', 'button');
            }
        });

        images = imgs.map(function (img) {
            // 优先使用原图（data-src / href），否则用当前 src
            var src = img.dataset.src || img.currentSrc || img.src;
            return { src: src, alt: img.alt || img.title || '' };
        });
    }
    function init() {
        collectImages();
        if (clickBound || !images.length) return;

        var container = document.querySelector('.post-content') || document.querySelector('.index-middle');
        if (!container) return;

        clickBound = true;
        container.addEventListener('keydown', function (e) {
            var target = e.target;
            if (target.tagName !== 'IMG' || (e.key !== 'Enter' && e.key !== ' ')) return;
            var idx = imgs.indexOf(target);
            if (idx >= 0) {
                e.preventDefault();
                open(idx);
            }
        });
        container.addEventListener('click', function (e) {
            var target = e.target;
            // Enter on a link that wraps a gallery image clicks the link, not
            // the image; open the viewer for keyboard users too.
            var link = target.tagName === 'A' ? target : null;
            if (link && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey) {
                var inner = link.querySelector('img');
                if (inner && imgs.indexOf(inner) >= 0 && link.textContent.trim() === '') target = inner;
            }
            if (target.tagName === 'IMG' && !target.closest('.donate-container')) {
                var idx = parseInt(target.dataset.lbIndex, 10);
                if (isNaN(idx)) {
                    idx = imgs.indexOf(target);
                }
                if (idx >= 0) {
                    e.preventDefault();
                    open(idx);
                }
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // 加密文章解密后重新收集图片（点击委托只绑定一次）
    window.addEventListener('hexo-blog-decrypt', function () {
        init();
    });
}();
