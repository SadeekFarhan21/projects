/*
 * Post lists paged to the screen (the home page and the archive): every post
 * is in the page, and a page holds as many posts as fit in the window below
 * the list's top (at least 3), so a phone may show 3 per page and a large
 * screen all of them. data-post-list names the item selector (default
 * .post-preview); a [data-post-group] (an archive year) whose posts are all
 * on other pages is hidden with them.
 * The page is kept in the URL (#page-2) so Back and shared links work, and
 * the size is recomputed on resize, keeping the first visible post in view.
 * Without JavaScript every post is visible.
 */
(function () {
  var list = document.querySelector('[data-post-list]')
  var nav = document.querySelector('.post-list-pager')
  if (!list || !nav) return
  var items = Array.prototype.slice.call(
    list.querySelectorAll(list.getAttribute('data-post-list') || '.post-preview'))
  var groups = Array.prototype.slice.call(list.querySelectorAll('[data-post-group]'))
  if (!items.length) return
  var newer = nav.querySelector('[data-page="newer"]')
  var older = nav.querySelector('[data-page="older"]')
  var status = nav.querySelector('.post-list-status')

  var MIN = 3
  var size = MIN
  var page = 1

  // Posts differ in height (titles and tags wrap), so fit against the real
  // layout: the largest first page whose last post and pager both end inside
  // the window without scrolling.
  function bottomOf(el) {
    return el.getBoundingClientRect().bottom + window.scrollY
  }

  function fits(k) {
    size = k
    page = 1
    render(false)
    var end = nav.hidden ? bottomOf(items[Math.min(k, items.length) - 1]) : bottomOf(nav)
    return end <= window.innerHeight - 8
  }

  function measure() {
    var best = MIN
    for (var k = MIN; k <= items.length; k++) {
      if (!fits(k)) break
      best = k
    }
    return best
  }

  function pages() {
    return Math.max(1, Math.ceil(items.length / size))
  }

  function pageFromHash() {
    var m = /^#page-(\d+)$/.exec(window.location.hash)
    return m ? parseInt(m[1], 10) : 1
  }

  function render(moveFocus) {
    page = Math.min(Math.max(1, page), pages())
    var start = (page - 1) * size
    items.forEach(function (item, i) {
      item.hidden = i < start || i >= start + size
    })
    groups.forEach(function (group) {
      group.hidden = !group.querySelector('li:not([hidden])')
    })
    nav.hidden = pages() === 1
    newer.disabled = page === 1
    older.disabled = page === pages()
    status.textContent = 'Page ' + page + ' of ' + pages()
    if (moveFocus) {
      var link = items[start] && items[start].querySelector('a')
      if (link) link.focus({ preventScroll: true })
      list.scrollIntoView({ block: 'start' })
    }
  }

  function go(to) {
    page = to
    var hash = page > 1 ? '#page-' + page : ' '
    history.pushState(null, '', hash === ' ' ? window.location.pathname : hash)
    render(true)
  }

  newer.addEventListener('click', function () { go(page - 1) })
  older.addEventListener('click', function () { go(page + 1) })
  window.addEventListener('popstate', function () {
    page = pageFromHash()
    render(false)
  })

  var resizeTimer
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer)
    resizeTimer = setTimeout(function () {
      var first = (page - 1) * size
      size = measure()
      page = Math.floor(first / size) + 1
      render(false)
    }, 150)
  })

  function init() {
    size = measure()
    page = pageFromHash()
    render(false)
  }

  init()
  // The web fonts change text heights, so fit again once they have loaded.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(init)
})()
