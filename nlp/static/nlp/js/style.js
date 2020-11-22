var __ERROR__ = '__ERROR__'
var __SUCCESS__ = '__SUCCESS__'

$(function() {
  // Initialization
  $('[data-toggle="tooltip"]').tooltip()
  new WOW().init()

  // Browser Detect
  if (bowser.name == 'Internet Explorer') {
    raise_modal_error('不兼容当前浏览器！')
    return
  }
})

function ajax_src_submit(source, qtype) {
  // Disable Operation
  disable_operation(qtype)

  // Print Source
  // console.log('source:', source)

  var flag = __SUCCESS__
  var qurl = '../query_' + qtype + '/'
  $.ajax({
    type: 'post',
    url: qurl,
    data: {
      'source': source,
      csrfmiddlewaretoken: $('[name="csrfmiddlewaretoken"]').val()
    },
    retry_count: 0,
    retry_limit: 3,
    success: function(ret) {
      // Print Result
      // console.log('result:', ret)

      try {
        switch (qtype) {
          case 'extract':
            flag = parse_extract(ret['jextract'])
            retry_ajax_submit(flag, this)
            break
          case 'contrast':
            if (ret['jcontrast'] == __ERROR__) {
              raise_modal_error('未知错误，请重试！')
              return
            }
            break
          default:
            raise_modal_error('未知错误，请重试！')
            break
        }
      } catch (e) {
        raise_modal_error('未知错误，请重试！')
        console.error(e)
      } finally {
        if (flag == __ERROR__ && this.retry_count <= this.retry_limit) {
          return
        }
        enable_operation(qtype)
      }
    },
    error: function(ret) {
      console.error(ret)
      enable_operation(qtype)
    }
  })
}

// Operation
function disable_operation(qtype) {
  switch (qtype) {
    case 'extract':
      // Mask
      $('#mask_extract_wait').fadeIn()
      // Button
      $('#extract_button').html('<div \
        class="spinner-border spinner-border-sm mr-1" \
        role="status" aria-hidden="true"></div>' + '抽取中...').addClass('disabled')
      $('#upload_button').html('<div \
        class="spinner-border spinner-border-sm mr-1" \
        role="status" aria-hidden="true"></div>' + '上传中...').addClass('disabled')
      $('#export_button').addClass('disabled')
      break
    default:
      break
  }
}

function enable_operation(qtype) {
  switch (qtype) {
    case 'extract':
      // Mask
      $('#mask_extract_wait').fadeOut()
      // Button
      $('#extract_button').html('开始抽取<i \
        class="fas fa-arrow-right ml-1"></i>').removeClass('disabled')
      $('#upload_button').html('<i \
        class="fas fa-arrow-up mr-1"></i>上传文档').removeClass('disabled')
      $('#export_button').removeClass('disabled')
      break
    default:
      break
  }
}

// Contrast
function toggle_contrast() {
  var $contrast = $('#switch_contrast input')[0]
  var s_con, t_con, s_nav, t_nav, t_ckb
  if ($contrast.checked) {
    [s_con, t_con, s_nav, t_nav, t_ckb] = ['moon', 'sun', 'dark', 'light', 'unchecked']
  } else {
    [s_con, t_con, s_nav, t_nav, t_ckb] = ['sun', 'moon', 'light', 'dark', 'checked']
  }

  $contrast.checked = !$contrast.checked
  $('.navbar').removeClass('navbar-' + s_nav).addClass('navbar-' + t_nav)
  $('.' + s_con).each(function() {
    $(this).removeClass(s_con).addClass(t_con)
  })

  ajax_src_submit({
    'contrast': t_con,
    'navbar': t_nav,
    'checkbox': t_ckb,
  }, 'contrast')
}

function get_contrast() {
  var $contrast = $('#switch_contrast input')[0]
  if ($contrast.checked) {
    return 'moon'
  } else {
    return 'sun'
  }
}

// Is Empty
function is_empty(obj) {
  return (typeof obj === 'undefined' || obj == null || obj == '' || obj.length == 0)
}

// Punctuation
function is_punctuation(s) {
  var punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
  return punctuation.includes(s)
}

function is_right_punctuation(s) {
  var punctuation = '!"#$%&\')*+,-./:;<=>?@\\]^_`|}~'
  return punctuation.includes(s)
}

function is_left_punctuation(s) {
  var punctuation = '([{'
  return punctuation.includes(s)
}

// Error
function retry_ajax_submit(flag, $ajax) {
  if (flag == __ERROR__) {
    $ajax.retry_count++
    if ($ajax.retry_count <= $ajax.retry_limit) {
      console.error('获取预测结果失败，重试中... (' + $ajax.retry_count + '/' + $ajax.retry_limit + ')')
      $('.mask-wait h6').html('请耐心等待，重试中...' + ' (' + $ajax.retry_count + '/' + $ajax.retry_limit + ')')
      $.ajax($ajax)
      return __ERROR__
    } else {
      raise_modal_error('获取预测结果失败！')
      return __ERROR__
    }
  }
  return __SUCCESS__
}

function raise_modal_error(error_info) {
  $('#modal_error #modal_error_content').text(error_info)
  $('#modal_error').modal()
  console.error(error_info)
}

// Resize
$.event.special.widthChanged = {
  remove: function() {
    $(this).children('iframe.width-changed').remove()
  },
  add: function() {
    var elm = $(this)
    var iframe = elm.children('iframe.width-changed')
    if (!iframe.length) {
      iframe = $('<iframe/>').addClass('width-changed').prependTo(this)
    }
    var oldWidth = elm.width()

    function elmResized() {
      var width = elm.width()
      if (oldWidth != width) {
        elm.trigger('widthChanged', [width, oldWidth])
        oldWidth = width
      }
    }

    var timer = 0;
    var ielm = iframe[0];

    (ielm.contentWindow || ielm).onresize = function() {
      clearTimeout(timer)
      timer = setTimeout(elmResized, 20)
    };
  }
}

// Debouncing
var waitForFinalEvent = (function() {
  var timers = {}
  return function(callback, ms, uniqueId) {
    if (!uniqueId) {
      uniqueId = "Don't call this twice without a uniqueId."
    }
    if (timers[uniqueId]) {
      clearTimeout(timers[uniqueId])
    }
    timers[uniqueId] = setTimeout(callback, ms)
  }
})()

// Progress
NProgress.configure({
  showSpinner: false
})

$(document).ajaxStart(function() {
  NProgress.start()
})

$(document).ajaxStop(function() {
  NProgress.done()
})
