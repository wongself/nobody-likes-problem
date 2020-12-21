var max_length = 5000
var pre_mrc_result = null

$(function() {
  // Initialization
  $('#left_text_area').val('')

  // Scrollbar
  new SimpleBar($('#right_result_container')[0])

  // Key Press
  $(document).on('keydown', function(e) {
    // mrc
    if (e.key == 'Enter' && e.ctrlKey) {
      $('#mrc_button').click()
    }
  })

  // Placeholder
  $('#left_text_area').on('input propertychange', function() {
    toggle_textarea_placeholder()
  })

  // Export
  $('#export_button').on('click', function() {
    export_result(pre_mrc_result)
  })
})

// mrc
function trigger_mrc() {
  var $src = $('#left_text_area')
  var src = $src.val()

  src = src.replace(/(^\s*)|(\s*$)/g, '')
  // src = src.replace(/\s+/g, ' ')
  $src.val(src)

  if (src.length <= 0) {
    raise_modal_error('无有效输入！')
    return
  }

  if (src.indexOf('\n') == src.length - 1 || src.indexOf('\n') == -1 || src.slice(0, src.indexOf('\n')).length < 0 || src.slice(src.indexOf('\n') + 1).length < 0) {
    raise_modal_error('参照下方描述来输入问题和文章！')
    return
  }

  ajax_src_submit(src, 'mrc')
}

function parse_mrc(jresult) {
  if (is_empty(jresult) || jresult == __ERROR__) {
    return __ERROR__
  }
  pre_mrc_result = jresult
  $('#right_result_area').html(jresult)

  toggle_result_placeholder()
}
