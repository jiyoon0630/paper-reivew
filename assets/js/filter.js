// Client-side tag filtering for the paper list on the home page.
document.addEventListener('DOMContentLoaded', function () {
  var buttons = document.querySelectorAll('.filter-btn');
  var items = document.querySelectorAll('.paper-item');
  if (!buttons.length) return;

  buttons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var tag = btn.getAttribute('data-tag');
      buttons.forEach(function (b) { b.classList.remove('active'); });
      btn.classList.add('active');

      items.forEach(function (item) {
        var tags = (item.getAttribute('data-tags') || '').trim().split(/\s+/);
        var show = tag === 'all' || tags.indexOf(tag) !== -1;
        item.style.display = show ? '' : 'none';
      });
    });
  });
});
