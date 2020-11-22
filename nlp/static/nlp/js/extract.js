var max_length = 5000
var pre_extract_result = null

$(function() {
  // Initialization
  $('#left_text_area').val('')
  $('[data-toggle="tooltip"]').tooltip()
  // $('.pane-mask').show()
  new WOW().init()

  // Browser Detect
  if (bowser.name == 'Internet Explorer') {
    raise_modal_error('不兼容当前浏览器！')
    return
  }

  // Resize
  $('#right_result_container').on('widthChanged', function() {
    waitForFinalEvent(function() {
      if ($('#right_result_area .showcase-sentence').length > 0 && pre_extract_result != null) {
        // console.log('Result can be rerended.')
        $('#render_result_button').fadeIn()
      }
    }, 500, 'pdf_render')
  })

  // Scrollbar
  new SimpleBar($('#right_result_container')[0])

  // Key Press
  $(document).on('keydown', function(e) {
    // Extract
    if (e.key == 'Enter' && e.ctrlKey) {
      $('#extract_button').click()
    }
    // Upload
    if (e.key == 'u' && e.ctrlKey && e.altKey) {
      $('#upload_button').click()
    }
  })

  // Placeholder
  $('#left_text_area').on('input propertychange', function() {
    toggle_textarea_placeholder()
  })

  // Upload
  $('#upload_button').on('click', function() {
    $('#upload_input').click()
  })

  $('#upload_input').on('change', function(e) {
    upload_document(e.target.files[0])
  })

  // Export
  $('#export_button').on('click', function() {
    export_result(pre_extract_result)
  })
})

// Extract
function trigger_extract() {
  $('.render-button').hide()

  var $src = $('#left_text_area')
  var src = $src.val()

  src = src.replace(/(^\s*)|(\s*$)/g, '')
  src = src.replace(/\s+/g, ' ')
  $src.val(src)

  if (src.length <= 0) {
    raise_modal_error('无有效输入！')
    return
  }

  ajax_src_submit(src, 'extract')
}

function parse_extract(jresult) {
  if (is_empty(jresult) || jresult == __ERROR__) {
    return __ERROR__
  }
  pre_extract_result = jresult
  annotate_predict(jresult, $('#right_result_area'))
  toggle_result_placeholder()
}

// Placeholder
function toggle_textarea_placeholder() {
  var $left_area = $('#left_text_area')
  var $left_stat_curr = $('#left_text_stat_curr')
  var $left_place = $('#left_text_place')
  var left_text_length = $left_area.val().length
  var left_text_remain = max_length - left_text_length

  if (left_text_length > 0) {
    $left_place.hide()
  } else {
    $left_place.show()
  }

  if (left_text_remain > 0) {
    $left_stat_curr.html(left_text_length)
  } else {
    $left_stat_curr.html(max_length)
    $left_area.val($left_area.val().substring(0, max_length))
  }
}

function toggle_result_placeholder() {
  var $tgt = $('#right_result_area')
  if (is_empty($tgt.html())) {
    $('#right_result_place').show()
  } else {
    $('#right_result_place').hide()
  }
}

// Render
function _re_render_result() {
  $('#render_result_button').fadeOut()
  annotate_predict(pre_extract_result, $('#right_result_area'))
}
