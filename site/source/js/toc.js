/*
 * Table of contents behaviour for post pages: the right-hand #toc and the
 * collapsible #toc-mobile. Replaces the theme's scroll-spy, which looked for
 * the list as the TOC's first child (it is the "Contents" heading) and so
 * never highlighted anything.
 *
 * - highlights the section currently being read
 * - smooth-scrolls to a heading on click and updates the URL hash
 * - keeps the active entry visible inside the sidebar
 * - closes the mobile TOC after a pick
 * - keeps sub-lists collapsed except the branch holding the current entry
 */
(function () {
  var tocs = Array.prototype.slice.call(document.querySelectorAll('#toc, #toc-mobile'))
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

  function current() {
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

  var ticking = false
  function onScroll() {
    if (ticking) return
    ticking = true
    requestAnimationFrame(function () { ticking = false; mark() })
  }
  window.addEventListener('scroll', onScroll, { passive: true })
  window.addEventListener('resize', onScroll)

  function closeMobile() {
    var m = document.getElementById('toc-mobile')
    if (!m || !m.classList.contains('is-open')) return
    m.classList.remove('is-open')
    var btn = m.querySelector('.toc-mobile-toggle')
    if (btn) { btn.setAttribute('aria-expanded', 'false'); btn.textContent = 'Show' }
  }

  links.forEach(function (a) {
    a.addEventListener('click', function (e) {
      var el = document.getElementById(idOf(a))
      if (!el) return
      e.preventDefault()
      closeMobile()
      var y = el.getBoundingClientRect().top + window.scrollY - TOP
      window.scrollTo({ top: y, behavior: reduceMotion ? 'auto' : 'smooth' })
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
