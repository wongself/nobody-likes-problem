var max_length = 5000
var pre_dialog_result = null

$(function() {
  // Initialization
  $('#left_text_area').val('')

  // Scrollbar
  new SimpleBar($('#right_result_container')[0])

  // Key Press
  $(document).on('keydown', function(e) {
    // dialog
    if (e.key == 'Enter' && e.ctrlKey) {
      $('#dialog_button').click()
    }
  })

  // Placeholder
  $('#left_text_area').on('input propertychange', function() {
    toggle_textarea_placeholder()
  })

  // Export
  $('#export_button').on('click', function() {
    export_result(pre_dialog_result)
  })
})

// dialog
function trigger_dialog() {
  var $src = $('#left_text_area')
  var src = $src.val()

  src = src.replace(/(^\s*)|(\s*$)/g, '')
  src = src.replace(/\s+/g, ' ')
  $src.val(src)

  if (src.length <= 0) {
    raise_modal_error('无有效输入！')
    return
  }

  ajax_src_submit(src, 'dialog')
  $('#right_result_area').append('\
    <div class="card canvas-bgcolor ' + $('#input_contrast').val() + ' mb-3">\
      <div class="card-body p-3">\
        <h5 class="card-title stress-color ' + $('#input_contrast').val() + ' mb-2">问题</h5>\
        <p class="card-text oppost-color ' + $('#input_contrast').val() + '">' + src + '</p>\
      </div>\
    </div>')
}

function parse_dialog(jresult) {
  if (is_empty(jresult) || jresult == __ERROR__) {
    return __ERROR__
  }
  pre_dialog_result = jresult
  $('#right_result_area').append('\
    <div class="card canvas-bgcolor ' + $('#input_contrast').val() + ' mb-3">\
      <div class="card-body p-3">\
        <h5 class="card-title stress-color ' + $('#input_contrast').val() + ' text-right mb-2">回答</h5>\
        <p class="card-text oppost-color ' + $('#input_contrast').val() + ' text-right">' + jresult + '</p>\
      </div>\
    </div>')

  toggle_result_placeholder()
}
