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
          case 'contrast':
            if (ret['jcontrast'] == __ERROR__) {
              raise_modal_error('未知错误，请重试！')
              return
            }
            break
          case 'extract':
            flag = parse_extract(ret['jextract'])
            retry_ajax_submit(flag, this)
            break
<<<<<<< HEAD
          case 'dialog': 
            flag = parse_dialog(ret['jdialog'])
=======
          case 'text_classification_ch':
            flag = parse_text_classification_ch(ret['jtext_classification_ch'])
            retry_ajax_submit(flag, this)
            break
          case 'translation':
            flag = parse_translation(ret['jtranslation'])
            retry_ajax_submit(flag, this)
            break
          case 'mrc':
            flag = parse_mrc(ret['jmrc'])
            retry_ajax_submit(flag, this)
            break
          case 'sana':
            flag = parse_sana(ret['jsana'])
>>>>>>> fdfb8794cdbd4b8acc74bf89373da7ab52c41164
            retry_ajax_submit(flag, this)
            break
          case 'template': // 复制该段，粘贴在该段之上，并将 template 字段进行重命名，就像 case 'extract' 一样。
            flag = parse_template(ret['jtemplate'])
            retry_ajax_submit(flag, this)
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
      // Button
      $('#extract_button').html('<div \
        class="spinner-border spinner-border-sm mr-1" \
        role="status" aria-hidden="true"></div>' + '抽取中...').addClass('disabled')
      $('#upload_button').html('<div \
        class="spinner-border spinner-border-sm mr-1" \
        role="status" aria-hidden="true"></div>' + '上传中...').addClass('disabled')
<<<<<<< HEAD
    case 'dialog': 
      // Button
      $('#dialog_button').html('<div \
       class="spinner-border spinner-border-sm mr-1" \
       role="status" aria-hidden="true"></div>' + '搜索中...').addClass('disabled')
=======
    case 'text_classification_ch':
      // Button
      $('#text_classification_ch_button').html('<div \
        class="spinner-border spinner-border-sm mr-1" \
        role="status" aria-hidden="true"></div>' + '分类中...').addClass('disabled')
    case 'translation':
      // Button
      $('#translation_button').html('<div \
        class="spinner-border spinner-border-sm mr-1" \
        role="status" aria-hidden="true"></div>' + '翻译中...').addClass('disabled')
    case 'mrc':
      // Button
      $('#mrc_button').html('<div \
        class="spinner-border spinner-border-sm mr-1" \
        role="status" aria-hidden="true"></div>' + '阅读中...').addClass('disabled')
    case 'sana':
      // Button
      $('#sana_button').html('<div \
       class="spinner-border spinner-border-sm mr-1" \
       role="status" aria-hidden="true"></div>' + '分析中...').addClass('disabled')
>>>>>>> fdfb8794cdbd4b8acc74bf89373da7ab52c41164
    case 'template': // 复制该段，粘贴在该段之上，并将 template 字段进行重命名，就像 case 'extract' 一样。
      // Button
      $('#template_button').html('<div \
       class="spinner-border spinner-border-sm mr-1" \
       role="status" aria-hidden="true"></div>' + '搜索中...').addClass('disabled')
    default:
      // Mask
      $('#mask_result_wait').fadeIn()
      // Button
      $('#export_button').addClass('disabled')
      break
  }
}

function enable_operation(qtype) {
  switch (qtype) {
    case 'extract':
      // Button
      $('#extract_button').html('开始抽取<i \
        class="fas fa-arrow-right ml-1"></i>').removeClass('disabled')
      $('#upload_button').html('<i \
        class="fas fa-arrow-up mr-1"></i>上传文档').removeClass('disabled')
<<<<<<< HEAD
    case 'dialog':// 复制该段，粘贴在该段之上，并将 template 字段进行重命名，就像 case 'extract' 一样。
      // Button
      $('#dialog_button').html('开始对话<i \
=======
    case 'text_classification_ch':
      // Button
      $('#text_classification_ch_button').html('开始分类<i \
        class="fas fa-arrow-right ml-1"></i>').removeClass('disabled')
    case 'translation':
      // Button
      $('#translation_button').html('开始翻译<i \
        class="fas fa-arrow-right ml-1"></i>').removeClass('disabled')
    case 'mrc':
      // Button
      $('#mrc_button').html('开始阅读<i \
        class="fas fa-arrow-right ml-1"></i>').removeClass('disabled')
    case 'sana':
      // Button
      $('#sana_button').html('开始分析<i \
>>>>>>> fdfb8794cdbd4b8acc74bf89373da7ab52c41164
        class="fas fa-arrow-right ml-1"></i>').removeClass('disabled')
    case 'template': // 复制该段，粘贴在该段之上，并将 template 字段进行重命名，就像 case 'extract' 一样。
      // Button
      $('#template_button').html('开始抽取<i \
        class="fas fa-arrow-right ml-1"></i>').removeClass('disabled')
    default:
      // Mask
      $('#mask_result_wait').fadeOut()
      // Button
      $('#export_button').removeClass('disabled')
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

// Export Predictions
function export_result(result) {
  if (result == null) {
    raise_modal_error('没有预测可导出！')
    return
  }

  var jexport = JSON.stringify(result)
  var blob = new Blob([jexport], {
    type: 'text/plain;charset=utf-8'
  })
  var filename = 'result_of_extract ' + generate_timestamp() + '.json'

  var url = window.URL || window.webkitURL
  link = url.createObjectURL(blob)
  var a = $('<a />')
  a.attr('download', filename)
  a.attr('href', link)
  $('body').append(a)
  a[0].click()
  $('body').remove(a)
}

function generate_timestamp() {
  var curr_time = new Date().Format('yyyy-MM-dd hh_mm_ss')
  return curr_time
}

Date.prototype.Format = function(fmt) {
  var o = {
    'M+': this.getMonth() + 1,
    'd+': this.getDate(),
    'h+': this.getHours(),
    'm+': this.getMinutes(),
    's+': this.getSeconds(),
    'q+': Math.floor((this.getMonth() + 3) / 3),
    'S': this.getMilliseconds()
  }
  if (/(y+)/.test(fmt)) fmt = fmt.replace(RegExp.$1, (this.getFullYear() + '').substr(4 - RegExp.$1.length))
  for (var k in o)
    if (new RegExp('(' + k + ')').test(fmt)) fmt = fmt.replace(RegExp.$1, (RegExp.$1.length == 1) ? (o[k]) : (('00' + o[k]).substr(('' + o[k]).length)))
  return fmt
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
