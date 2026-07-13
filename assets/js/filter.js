// Home page: combined language + tag filtering for the paper list.
document.addEventListener('DOMContentLoaded', function () {
  var langButtons = document.querySelectorAll('.lang-btn');
  var tagButtons = document.querySelectorAll('.filter-btn');
  var items = document.querySelectorAll('.paper-item');
  var emptyLang = document.getElementById('empty-lang');
  if (!items.length) return;

  var curLang = 'en'; // default: English-first
  var curTag = 'all';

  function setActive(buttons, btn) {
    buttons.forEach(function (b) { b.classList.remove('active'); });
    btn.classList.add('active');
  }

  function apply() {
    var visible = 0;
    items.forEach(function (item) {
      var lang = item.getAttribute('data-lang');
      var tags = (item.getAttribute('data-tags') || '').trim().split(/\s+/);
      var show = lang === curLang && (curTag === 'all' || tags.indexOf(curTag) !== -1);
      item.style.display = show ? '' : 'none';
      if (show) visible++;
    });
    if (emptyLang) emptyLang.hidden = visible !== 0;
  }

  langButtons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      curLang = btn.getAttribute('data-lang');
      setActive(langButtons, btn);
      apply();
    });
  });

  tagButtons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      curTag = btn.getAttribute('data-tag');
      setActive(tagButtons, btn);
      apply();
    });
  });

  apply();
});
