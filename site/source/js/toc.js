/*
 * Table of contents behaviour for post pages: the right-hand #toc sidebar
 * (shown above 1180px). Replaces the theme's scroll-spy, which looked for
 * the list as the TOC's first child (it is the "Contents" heading) and so
 * never highlighted anything.
 *
 * - highlights the section currently being read
 * - smooth-scrolls to a heading on click and updates the URL hash
 * - keeps the active entry visible inside the sidebar
 * - keeps sub-lists collapsed except the branch holding the current entry
 */
(function () {
  var tocs = Array.prototype.slice.call(document.querySelectorAll('#toc'))
  if (!tocs.length) return

  var links = []
  tocs.forEach(function (t) {
    links = links.concat(Array.prototype.slice.call(t.querySelectorAll('a[href^="#"]')))
  })
  if (!links.length) return

  var idOf = function (a) {
    try { return decodeURIComponent(a.getAttribute('href').slice(1)) } catch (e) { return a.getAttribute('href').slice(1) }
  }
  var seen = {}
  var headings = links
    .map(idOf)
    .filter(function (id) { if (seen[id]) return false; seen[id] = 1; return true })
    .map(function (id) { return document.getElementById(id) })
    .filter(Boolean)
    .sort(function (a, b) { return a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1 })
  if (!headings.length) return

  var TOP = 24 // where a heading lands after a jump
  var READ_LINE = 120 // a section is "current" once its heading passes this line
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  var activeId = null

  // A TOC click pins the clicked heading as current. Otherwise a section whose
  // first subsection sits within READ_LINE of it would light up that
  // subsection instead of the entry that was clicked. The pin holds while the
  // jump is in flight and, once the scroll settles, until the heading is more
  // than PIN_SLACK px from where the jump puts it (so it also lets go when the
  // reader scrolls before the jump has finished).
  var PIN_SLACK = 16
  var pin = null // { el, land: viewport top the jump puts el at, settled }
  var settleTimer = 0
  function settle() {
    clearTimeout(settleTimer)
    settleTimer = setTimeout(function () {
      if (!pin || pin.settled) return
      pin.settled = true
      activeId = null
      mark()
    }, 150)
  }

  function current() {
    if (pin) {
      if (!pin.settled) return pin.el.id
      if (Math.abs(pin.el.getBoundingClientRect().top - pin.land) <= PIN_SLACK) return pin.el.id
      pin = null
    }
    var atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4
    if (atBottom) return headings[headings.length - 1].id
    var id = headings[0].id
    for (var i = 0; i < headings.length; i++) {
      if (headings[i].getBoundingClientRect().top - READ_LINE <= 0) id = headings[i].id
      else break
    }
    return id
  }

  function keepVisible(a) {
    var box = a.closest('#toc')
    if (!box || box.scrollHeight <= box.clientHeight) return
    var r = a.getBoundingClientRect()
    var b = box.getBoundingClientRect()
    if (r.top < b.top + 40) box.scrollTop -= b.top + 40 - r.top
    else if (r.bottom > b.bottom - 24) box.scrollTop += r.bottom - (b.bottom - 24)
  }

  // Open only the sub-lists on the path to the active entry; collapse the rest.
  // The section that contains the active entry is marked too (.is-trail), so
  // both the current section and the current subsection read as bold.
  function expandBranch() {
    tocs.forEach(function (t) {
      Array.prototype.forEach.call(t.querySelectorAll('a.is-trail'), function (a) { a.classList.remove('is-trail') })
      var active = t.querySelector('a.is-active')
      for (var li = active && active.parentElement ? active.parentElement.parentElement.closest('li') : null; li; li = li.parentElement.closest('li')) {
        var head = li.querySelector(':scope > a')
        if (head) head.classList.add('is-trail')
      }
      var branches = t.querySelectorAll('li')
      Array.prototype.forEach.call(branches, function (li) {
        if (!li.querySelector(':scope > ol, :scope > ul')) return
        li.classList.toggle('is-open', !!li.querySelector('a.is-active'))
      })
    })
  }

  function mark() {
    var id = current()
    if (id === activeId) return
    activeId = id
    links.forEach(function (a) {
      var on = idOf(a) === id
      var li = a.parentElement
      if (li) li.classList.toggle('active', on)
      a.classList.toggle('is-active', on)
      if (on) {
        a.setAttribute('aria-current', 'location')
        keepVisible(a)
      } else {
        a.removeAttribute('aria-current')
      }
    })
    expandBranch()
    links.forEach(function (a) { if (a.classList.contains('is-active')) keepVisible(a) })
  }

  // Collapsed sub-lists are display:none, so keyboard users could never tab
  // into another section's subsections. Open the branch of whichever entry
  // has focus, alongside the reading branch.
  tocs.forEach(function (t) {
    t.addEventListener('focusin', function (e) {
      var a = e.target.closest && e.target.closest('a')
      if (!a) return
      expandBranch()
      for (var li = a.closest('li'); li && t.contains(li); li = li.parentElement.closest('li')) {
        if (li.querySelector(':scope > ol, :scope > ul')) li.classList.add('is-open')
      }
    })
  })

  var ticking = false
  function onScroll() {
    if (pin && !pin.settled) settle()
    if (ticking) return
    ticking = true
    requestAnimationFrame(function () { ticking = false; mark() })
  }
  window.addEventListener('scroll', onScroll, { passive: true })
  window.addEventListener('resize', onScroll)

  links.forEach(function (a) {
    a.addEventListener('click', function (e) {
      // leave modified clicks (new tab / window) to the browser
      if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return
      var el = document.getElementById(idOf(a))
      if (!el) return
      e.preventDefault()
      var y = el.getBoundingClientRect().top + window.scrollY - TOP
      var maxY = document.documentElement.scrollHeight - window.innerHeight
      pin = { el: el, land: el.getBoundingClientRect().top + window.scrollY - Math.max(0, Math.min(y, maxY)), settled: false }
      settle() // in case the page is already there and no scroll event fires
      window.scrollTo({ top: y, behavior: reduceMotion ? 'auto' : 'smooth' })
      // move keyboard focus to the section, as the default jump would have
      if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '-1')
      el.focus({ preventScroll: true })
      if (history.replaceState) history.replaceState(null, '', '#' + encodeURIComponent(el.id))
      activeId = null
      // highlight the target now rather than waiting for the scroll to finish
      links.forEach(function (x) {
        var on = x === a || idOf(x) === el.id
        if (x.parentElement) x.parentElement.classList.toggle('active', on)
        x.classList.toggle('is-active', on)
      })
      activeId = el.id
      expandBranch()
    })
  })

  mark()
})()
