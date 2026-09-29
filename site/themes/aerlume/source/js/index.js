/**
 * Created by Xiaotao.Nie on 09/04/2018.
 * All right reserved
 */

function escapeHTML(value) {
    return String(value).replace(/[&<>"]/g, function (char) {
        return {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;'
        }[char]
    })
}

function escapeRegExp(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function highlightKeyword(value, keyword) {
    var escapedValue = escapeHTML(value)
    if (!keyword) return escapedValue

    var flags = caseSensitive ? 'g' : 'ig'
    return escapedValue.replace(new RegExp('(' + escapeRegExp(escapeHTML(keyword)) + ')', flags), "<span class='red'>$1</span>")
}

// Global functions and listeners
window.onresize = function () {
    if (window.document.documentElement.clientWidth > 680) {
        var aboutContent = document.getElementById('nav-content')
        if (aboutContent) {
            aboutContent.classList.remove('hide-block')
            aboutContent.classList.remove('show-block')
            aboutContent.classList.remove('is-mobile-open')
            aboutContent.hidden = true
        }
    }

    reHeightToc()
}

// Nav switch function on mobile
/*****************************************************************************/
var navToggle = document.getElementById('site-nav-toggle')
if (navToggle) {
    var navToggleButton = navToggle.querySelector('button')
    var aboutContent = document.getElementById('nav-content')

    function setMobileNavOpen(isOpen) {
        if (!aboutContent || !navToggleButton) return
        aboutContent.classList.toggle('is-mobile-open', isOpen)
        navToggle.classList.toggle('is-open', isOpen)
        aboutContent.hidden = !isOpen
        navToggleButton.setAttribute('aria-expanded', isOpen ? 'true' : 'false')
        navToggleButton.setAttribute('aria-label', isOpen ? 'Close site navigation' : 'Open site navigation')
    }

    if (navToggleButton && aboutContent) {
        aboutContent.hidden = true
        navToggleButton.addEventListener('click', function (event) {
            event.preventDefault()
            event.stopPropagation()
            setMobileNavOpen(!aboutContent.classList.contains('is-mobile-open'))
        })

        document.addEventListener('click', function (event) {
            if (window.innerWidth > 680 || !aboutContent.classList.contains('is-mobile-open')) return
            if (!aboutContent.contains(event.target)) setMobileNavOpen(false)
        })

        document.addEventListener('keydown', function (event) {
            if (event.key === 'Escape') setMobileNavOpen(false)
        })
    }

    document.querySelectorAll('#nav-content a').forEach(function (link) {
        link.addEventListener('click', function () {
            if (window.innerWidth > 680) return
            setMobileNavOpen(false)
        })
    })
}

var mobileToc = document.getElementById('toc-mobile')
var mobileTocToggle = mobileToc ? mobileToc.querySelector('.toc-mobile-toggle') : null
var mobileTocList = document.getElementById('toc-mobile-list')
if (mobileToc && mobileTocToggle && mobileTocList) {
    mobileTocToggle.addEventListener('click', function () {
        var isOpen = mobileToc.classList.toggle('is-open')
        mobileTocToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false')
        mobileTocToggle.textContent = isOpen ? 'Hide' : 'Show'
    })
}

// Desktop sidebar collapse/expand toggle
/*****************************************************************************/
var navToggleBtn = document.getElementById('nav-toggle-btn')
var navEl = document.getElementById('nav')
var sidebarEl = document.querySelector('.index-left')

function syncSidebarState() {
    if (!navEl || !sidebarEl) return

    var collapsed = navEl.classList.contains('is-collapsed')
    sidebarEl.classList.toggle('is-collapsed', collapsed)
    sidebarEl.classList.toggle('is-expanded', !collapsed)
}

if (navToggleBtn && navEl) {
    // 从 localStorage 恢复状态
    var savedNavState = localStorage.getItem('nav-collapsed')
    if (savedNavState === 'false') {
        navEl.classList.remove('is-collapsed')
        navEl.classList.add('is-expanded')
    }
    syncSidebarState()

    navToggleBtn.addEventListener('click', function () {
        var isCollapsed = navEl.classList.contains('is-collapsed')
        if (isCollapsed) {
            navEl.classList.remove('is-collapsed')
            navEl.classList.add('is-expanded')
            navToggleBtn.title = 'Collapse navigation'
            localStorage.setItem('nav-collapsed', 'false')
        } else {
            navEl.classList.remove('is-expanded')
            navEl.classList.add('is-collapsed')
            navToggleBtn.title = 'Expand navigation'
            localStorage.setItem('nav-collapsed', 'true')
        }
        syncSidebarState()
        // The sidebar TOC moves (or appears) when the nav opens or closes, so
        // tell the TOC code to re-measure where it sits.
        window.dispatchEvent(new Event('aerlume:nav-toggle'))
    })
}

// global search
/*****************************************************************************/
var searchButton = document.getElementById('search')
var searchField = document.getElementById('search-field')
var searchInput = document.getElementById('search-input')
var searchResultContainer = document.getElementById('search-result-container')
var escSearch = document.getElementById('esc-search')
var bgSearch = document.getElementById('search-bg')
var beginSearch = document.getElementById('begin-search')

var searchJson
var caseSensitive = false

if (searchButton && searchField && searchInput && searchResultContainer && escSearch && bgSearch && beginSearch) {
    searchField.addEventListener('mousewheel', function (e) {
        e.stopPropagation()
        return false
    }, false)

    searchButton.addEventListener('click', function () {
        search()
    })

    escSearch.addEventListener('click', function () {
        hideSearchField()
    })

    bgSearch.addEventListener('click', function () {
        hideSearchField()
    })

    beginSearch.addEventListener('click', function () {
        var keyword = searchInput.value
        if (keyword) {
            searchFromKeyWord(keyword)
        }
    })
}

function toggleSeachField() {
    if (!searchField) return

    if (!searchField.classList.contains('show-flex-fade')) {
        showSearchField()
    } else {
        hideSearchField()
    }
}

function showSearchField() {
    if (!searchField || !searchInput) return

    searchField.classList.add('show-flex-fade', 'search-animation')
    searchField.classList.remove('hide-flex-fade')
    window.setTimeout(function () { searchInput.focus() }, 0)
}

function hideSearchField() {
    if (!searchField) return

    window.onkeydown = null
    searchField.classList.add('hide-flex-fade')
    searchField.classList.remove('show-flex-fade')
    if (searchButton) searchButton.focus()
}

function searchFromKeyWord(keyword) {
    keyword = keyword || ''
    if (!searchResultContainer) return

    var result = []
    var slideWindowSize = 100
    var handleKeyword = caseSensitive ? keyword : keyword.toLowerCase()

    if (!searchJson) return -1

    searchJson.forEach(function (item) {
        if (!item.title || !item.content) return

        var title = String(item.title)
        var content = String(item.content).trim().replace(/<[^>]+>/g, '').replace(/[`#\n]/g, '')
        var lowerTitle = caseSensitive ? title : title.toLowerCase()
        var lowerContent = caseSensitive ? content : content.toLowerCase()

        if (lowerTitle.indexOf(handleKeyword) === -1 && lowerContent.indexOf(handleKeyword) === -1) return

        var resultItem = {
            title: highlightKeyword(title, keyword),
            url: item.url,
            content: []
        }

        var lastend = 0
        while (lowerContent.indexOf(handleKeyword) !== -1) {
            var keywordIndex = lowerContent.indexOf(handleKeyword)
            var begin = keywordIndex - slideWindowSize / 2 < 0 ? 0 : keywordIndex - slideWindowSize / 2
            var end = begin + slideWindowSize

            resultItem.content.push('...' + highlightKeyword(content.slice(lastend + begin, lastend + end), keyword) + '...')
            lowerContent = lowerContent.slice(end, lowerContent.length)
            lastend += end
        }

        result.push(resultItem)
    })

    if (!result.length) {
        searchResultContainer.innerHTML = '<div class="no-search-result">No matching posts</div>'
        return
    }

    var searchFragment = document.createElement('ul')

    result.forEach(function (item) {
        var searchItem = document.createElement('li')
        var searchTitle = document.createElement('a')
        searchTitle.href = item.url
        searchTitle.innerHTML = item.title
        searchItem.appendChild(searchTitle)

        if (item.content.length) {
            var searchContentLiContainer = document.createElement('ul')
            item.content.forEach(function (citem) {
                var searchContentFragment = document.createElement('li')
                searchContentFragment.innerHTML = citem
                searchContentLiContainer.appendChild(searchContentFragment)
            })
            searchItem.appendChild(searchContentLiContainer)
        }

        searchFragment.appendChild(searchItem)
    })

    while (searchResultContainer.firstChild) {
        searchResultContainer.removeChild(searchResultContainer.firstChild)
    }
    searchResultContainer.appendChild(searchFragment)
}

function search() {
    if (!searchField || !searchInput) return

    toggleSeachField()

    window.onkeydown = function (e) {
        if (e.which === 27) {
            toggleSeachField()
        } else if (e.which === 13) {
            var keyword = searchInput.value
            if (keyword) {
                searchFromKeyWord(keyword)
            }
        }
    }

    if (!searchJson) {
        var isXml = false
        var search_path = window.hexo_search_path
        if (search_path.length === 0) {
            search_path = 'search.json'
        } else if (/xml$/i.test(search_path)) {
            isXml = true
        }
        var path = window.hexo_root + search_path
        fetch(path)
            .then(function (res) { return isXml ? res.text() : res.json() })
            .then(function (res) {
                if (isXml) {
                    var parser = new DOMParser()
                    var doc = parser.parseFromString(res, 'application/xml')
                    searchJson = Array.from(doc.querySelectorAll('entry')).map(function (entry) {
                        return {
                            title: entry.querySelector('title').textContent,
                            content: entry.querySelector('content').textContent,
                            url: entry.querySelector('url').textContent
                        }
                    })
                } else {
                    searchJson = res
                }
            })
            .catch(function (err) {
                console.error('Search load failed:', err)
            })
    }
}

// directory function in post pages
/*****************************************************************************/
function getDistanceOfLeft(obj) {
    var left = 0
    var top = 0
    while (obj) {
        left += obj.offsetLeft
        top += obj.offsetTop
        obj = obj.offsetParent
    }
    return {
        left: left,
        top: top
    }
}

var toc = document.getElementById('toc')
var tocToTop = toc ? getDistanceOfLeft(toc).top : 0

function reHeightToc() {
    if (toc) {
        toc.style.maxHeight = (document.documentElement.clientHeight - 10) + 'px'
        toc.style.overflowY = 'scroll'
    }
}

reHeightToc()

if (window.isPost && toc && toc.children && toc.children[0]) {
    var result = []
    var nameSet = new Set()

    if (toc.children[0].nodeName === 'OL') {
        var ol = Array.from(toc.children[0].children)

        function getArrayFromOl(ol) {
            var result = []

            ol.forEach(function (item) {
                var link = item.children[0]
                if (!link) return

                var value = link.getAttribute('href').replace(/^#/, '')
                nameSet.add(value)

                if (item.children.length === 1) {
                    result.push({
                        value: [value],
                        dom: item
                    })
                } else {
                    var childList = item.children[1]
                    var concatArray = childList ? getArrayFromOl(Array.from(childList.children)) : []
                    result.push({
                        value: [value].concat(concatArray.reduce(function (p, n) {
                            p = p.concat(n.value)
                            return p
                        }, [])),
                        dom: item
                    })
                    result = result.concat(concatArray)
                }
            })
            return result
        }

        result = getArrayFromOl(ol)
    }

    var nameArray = Array.from(nameSet)

    // Where the TOC sits in the page when it is not pinned. 0 means it is
    // hidden (collapsed nav) and must never be pinned, since pinning a hidden
    // TOC at 0 left it stuck over the nav once the nav was opened.
    function measureTocTop() {
        toc = document.getElementById('toc')
        if (!toc) return
        toc.classList.remove('toc-fixed')
        tocToTop = getComputedStyle(toc).display === 'none' ? 0 : getDistanceOfLeft(toc).top
    }

    function reLayout() {
        var scrollToTop = document.documentElement.scrollTop || window.pageYOffset
        if (tocToTop === 0) measureTocTop()
        if (!toc) return
        if (tocToTop > 0 && tocToTop <= scrollToTop + 10) {
            if (!toc.classList.contains('toc-fixed')) toc.classList.add('toc-fixed')
        } else if (toc.classList.contains('toc-fixed')) {
            toc.classList.remove('toc-fixed')
        }

        var minTop = 9999
        var minTopsValue = ''

        nameArray.forEach(function (item) {
            item = decodeURIComponent(item)
            var dom = document.getElementById(item) || document.getElementById(item.replace(/\s/g, ''))
            if (!dom) return

            var toTop = getDistanceOfLeft(dom).top - scrollToTop
            if (Math.abs(toTop) < minTop) {
                minTop = Math.abs(toTop)
                minTopsValue = item
            }
        })

        if (minTopsValue) {
            result.forEach(function (item) {
                if (item.value.indexOf(encodeURIComponent(minTopsValue)) !== -1) {
                    item.dom.classList.add('active')
                } else {
                    item.dom.classList.remove('active')
                }
            })
        }
    }

    measureTocTop()
    reLayout()

    window.addEventListener('scroll', function () {
        reLayout()
    })

    function remeasure() {
        measureTocTop()
        reLayout()
    }
    window.addEventListener('aerlume:nav-toggle', function () {
        remeasure()
        // measure again once the nav's width transition has settled
        setTimeout(remeasure, 400)
    })
    window.addEventListener('resize', remeasure)
}

// donate
/*****************************************************************************/
var donateButton = document.getElementById('donate-button')
var donateImgContainer = document.getElementById('donate-img-container')
var donateImg = document.getElementById('donate-img')

if (donateButton && donateImgContainer) {
    donateButton.addEventListener('click', function () {
        if (donateImgContainer.classList.contains('hide')) {
            donateImgContainer.classList.remove('hide')
        } else {
            donateImgContainer.classList.add('hide')
        }
    })

    if (donateImg && donateImg.dataset.src) {
        donateImg.src = donateImg.dataset.src
    }
}

// 加密文章解密成功后显示目录（hexo-blog-encrypt 触发的事件）
/*****************************************************************************/
window.addEventListener('hexo-blog-decrypt', function () {
    ;['toc', 'toc-mobile'].forEach(function (id) {
        var el = document.getElementById(id)
        if (el) {
            el.hidden = false
            el.classList.remove('toc-encrypted')
        }
    })
    reHeightToc()
})
