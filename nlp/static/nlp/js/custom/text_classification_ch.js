var max_length = 5000
var pre_text_classification_ch_result = null

$(function() {
  // Initialization
  $('#left_text_area').val('')

  // Scrollbar
  new SimpleBar($('#right_result_container')[0])

  // Key Press
  $(document).on('keydown', function(e) {
    // text_classification_ch
    if (e.key == 'Enter' && e.ctrlKey) {
      $('#text_classification_ch_button').click()
    }
  })

  // Placeholder
  $('#left_text_area').on('input propertychange', function() {
    toggle_textarea_placeholder()
  })

  // Export
  $('#export_button').on('click', function() {
    export_result(pre_text_classification_ch_result)
  })
})

// text_classification_ch
function trigger_text_classification_ch() {
  var $src = $('#left_text_area')
  var src = $src.val()

  src = src.replace(/(^\s*)|(\s*$)/g, '')
  src = src.replace(/\s+/g, ' ')
  $src.val(src)

  if (src.length <= 0) {
    raise_modal_error('无有效输入！')
    return
  }

  ajax_src_submit(src, 'text_classification_ch')
}

function parse_text_classification_ch(jresult) {
  if (is_empty(jresult) || jresult == __ERROR__) {
    return __ERROR__
  }
  pre_text_classification_ch_result = jresult
  $('#right_result_area').html(jresult)

  toggle_result_placeholder()
}
