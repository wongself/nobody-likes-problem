// Annotate Predictions
function annotate_predict(jsentences, $annot) {
  $annot.html('')

  $.each(jsentences, function(s_index, jsentence) {
    annotate_sentence(s_index, jsentence, $annot)
  })

  stylize_sentence_height($annot)
  $('[data-toggle="tooltip"]').tooltip()
}

function stylize_sentence_height($annot) {
  $annot.children('.showcase-sentence').each(function(index) {
    stylize_line_height($(this))
  })
}

function stylize_line_height($sent) {
  var $cur_head = $sent.children('span:first')
  var arr_tokens = new Array()

  if ($cur_head.length <= 0) {
    return
  }

  var cur_cnt = 0
  var max_cnt = 400

  var margin_bottom = 8
  var margin_error = 1e-1

  // console.log('head', $cur_head.html())
  while ($cur_head.next().length > 0 && cur_cnt < max_cnt) {
    while ($cur_head.next().length > 0 && Math.abs(get_ycenter($cur_head.next()) - get_ycenter($cur_head)) <= margin_error) {
      // console.log('leap', $cur_head.html())
      arr_tokens.push($cur_head)
      $cur_head = $cur_head.next()
    }
    // console.log('final', $cur_head.html())
    arr_tokens.push($cur_head)

    if ($cur_head[0].getClientRects().length > 1) {
      $cur_head.after('<span class="target-other"></span>')
      arr_tokens.push($cur_head)
    }

    var arr_above = new Array(arr_tokens.length).fill(0)
    var arr_below = new Array(arr_tokens.length).fill(0)

    $.each(arr_tokens, function(index, $token) {
      var above_height = 0
      var below_height = 0

      if ($token.hasClass('target-other')) {
        // Above
        above_height = $token.height() / 2
        // Below
        $token.children('.target-path').each(function() {
          below_height = Math.max($(this).attr('height').replace(/(^-)|(px$)/g, '') - 0, below_height)
        })
        below_height += ($token.children('.target-path').length > 0) ? $token.children('.target-path').eq(0).css('top').replace(/(^-)|(px$)/g, '') - $token.height() / 2 : $token.height() / 2
      } else if ($token.hasClass('target-token')) {
        // Above
        above_height = ($token.children('.target-badge').length > 0) ? $token.children('.target-badge').css('top').replace(/(^-)|(px$)/g, '') - 0 : 0
        above_height += $token.height() / 2
        // Below
        $token.children('.target-path').each(function() {
          below_height = Math.max($(this).attr('height').replace(/(^-)|(px$)/g, '') - 0, below_height)
        })
        below_height += ($token.children('.target-path').length > 0) ? $token.children('.target-path').eq(0).css('top').replace(/(^-)|(px$)/g, '') - $token.height() / 2 : $token.height() / 2
      }

      arr_above[index] = above_height
      arr_below[index] = below_height
    })

    var max_hgt = Math.max(...arr_above) + Math.max(...arr_below)
    var max_top = Math.max(...arr_above)

    // Print Array
    // console.log('arr_above', arr_above)
    // console.log('arr_below', arr_below)

    $.each(arr_tokens, function(index, $token) {
      $token.css({
        'line-height': '' + (max_hgt + margin_bottom) + 'px',
        'top': '' + -1 * (max_hgt / 2 - max_top + margin_bottom / 2) + 'px'
      })
    })

    if ($cur_head.next().length <= 0) {
      break
    }

    $cur_head = $cur_head.next()
    arr_tokens = new Array()
    cur_cnt++
  }
}

function annotate_sentence(s_index, jsentence, $annot) {
  var annot_id = $annot.attr('id')

  var jtokens = jsentence['tokens']
  var jentities = jsentence['entities']
  var jrelations = jsentence['relations']

  // Entity
  var pre_end = 0
  var str = ''

  str += '<div class="showcase-sentence" id="showcase_sentence_' + s_index + '">'

  $.each(jentities, function(e_index, jentity) {
    var e_type = jentity['type']
    var e_start = jentity['start']
    var e_end = jentity['end']

    // Not Token
    str += '<span class="target-other">'
    for (let i = pre_end; i < e_start; i++) {
      str += (i && !is_right_punctuation(jtokens[i]) && !is_left_punctuation(jtokens[i - 1])) ? ' ' : ''
      str += jtokens[i] + '</span><span class="target-other">'
    }
    str += (e_start) ? ' ' : ''
    str += '</span>'

    // Token
    var id_ent = s_index + '_' + e_index
    str += '<span class="target-token" \
      id="' + annot_id + '_token_' + id_ent + '" \
      onmouseenter="token_mouseentered(this)" \
      onmouseleave="token_mouseleaved(this)"><span \
      class="target-entity etype-bgcolor-' + e_type.toLowerCase() + ' \
      bright ' + get_contrast() + '" id="' + annot_id + '_entity_' + id_ent + '">'

    for (let i = e_start; i < e_end; i++) {
      str += (i && i != e_start && !is_right_punctuation(jtokens[i]) && !is_left_punctuation(jtokens[i - 1])) ? '&nbsp;' : ''
      str += jtokens[i]
    }

    str += '</span><span class="target-badge \
      etype-bgcolor-' + e_type.toLowerCase() + '" \
      id="' + annot_id + '_badge_' + id_ent + '" \
      data-toggle="tooltip" data-placement="top" \
      title="' + get_verbosed_etype(e_type) + '" \
      >' + get_shorted_etype(e_type) + '</span></span>'

    pre_end = e_end
  })

  // Not Token
  str += '<span class="target-other">'
  for (let i = pre_end; i < jtokens.length; i++) {
    str += (i && !is_right_punctuation(jtokens[i]) && !is_left_punctuation(jtokens[i - 1])) ? ' ' : ''
    str += jtokens[i] + '</span><span class="target-other">'
  }
  str += '</span>'

  // Append Sentences
  $annot.append(str + '<br class="target-br"></div>')

  // Relation
  var layer_rel = arrange_relation(jrelations)

  $.each(jrelations, function(r_index, jrelation) {
    var r_type = jrelation['type']
    var r_head = jrelation['head']
    var r_tail = jrelation['tail']

    var $tgt_token_head = $('#' + annot_id + '_token_' + s_index + '_' + r_head)
    var $tgt_token_tail = $('#' + annot_id + '_token_' + s_index + '_' + r_tail)

    var rel_id = s_index + '_' + r_index

    $tgt_token_head.append('<i \
      class="fas fa-caret-down target-caret" \
      id="' + annot_id + '_caret_head_' + rel_id + '"></i>')
    $tgt_token_tail.append('<i \
      class="fas fa-caret-down target-caret" \
      id="' + annot_id + '_caret_tail_' + rel_id + '"></i>')

    connect_relation(rel_id, r_type, layer_rel[r_index], $tgt_token_head, $tgt_token_tail, $annot)
  })
}

function arrange_relation(jrelations) {
  var length_rel = jrelations.length
  if (length_rel <= 0) {
    return null
  }

  var sorted_rel = get_relation_sorted(jrelations)
  var flag_rel = new Array(length_rel).fill(false)
  var layer_rel = new Array(length_rel).fill(0)

  var cur_cnt = 0
  var max_cnt = 5

  var pre_last = sorted_rel[0]['end']
  flag_rel[sorted_rel[0]['index']] = true
  layer_rel[sorted_rel[0]['index']] = cur_cnt

  while (!is_rel_all_arranged(flag_rel) && cur_cnt < max_cnt) {
    // Arrange Greedily
    for (let i = 1; i < length_rel; i++) {
      if (sorted_rel[i]['start'] >= pre_last && !flag_rel[sorted_rel[i]['index']]) {
        pre_last = sorted_rel[i]['end']
        flag_rel[sorted_rel[i]['index']] = true
        layer_rel[sorted_rel[i]['index']] = cur_cnt
      }
    }

    pre_last = 0
    cur_cnt++
  }

  $.each(flag_rel, function(fr_index, f_rel) {
    if (!f_rel) {
      // flag_rel[fr_index] = true
      layer_rel[fr_index] = max_cnt
    }
  })

  // Print Arrangement
  // console.log('flag_rel', flag_rel)
  // console.log('layer_rel', layer_rel)

  return layer_rel
}

function get_relation_sorted(jrelations) {
  var arr_rel = new Array()

  $.each(jrelations, function(fr_index, jrelation) {
    var fr_head = jrelation['head']
    var fr_tail = jrelation['tail']

    var fr_start = Math.min(fr_head, fr_tail)
    var fr_end = Math.max(fr_head, fr_tail)

    arr_rel.push({
      'index': fr_index,
      'start': fr_start,
      'end': fr_end
    })
  })

  arr_rel.sort(sort_relation_by_end)

  return arr_rel
}

function sort_relation_by_end(a, b) {
  return a['end'] - b['end']
}

function is_rel_all_arranged(flag_rel) {
  return !flag_rel.includes(false)
}

function connect_relation(rel_id, r_type, layer, $head, $tail, $annot) {
  var annot_id = $annot.attr('id')
  var tgt_extract_width = $annot.width()

  // Width
  var head_xcenter = get_xcenter($head)
  var tail_xcenter = get_xcenter($tail)

  // Height
  var head_ycenter = get_ycenter($head)
  var tail_ycenter = get_ycenter($tail)

  // Path
  var path_width = null
  var path_d = null
  var path_svg = null

  var path_stack = 22

  var path_base = 10
  var path_arc_toe = path_stack * layer + path_base

  var path_minhgt = 20
  var path_height = path_arc_toe + path_minhgt

  var margin_error = 1e-1

  if (Math.abs(head_ycenter - tail_ycenter) <= margin_error) {
    // Same Line
    path_width = Math.abs(head_xcenter - tail_xcenter)
    path_d = get_path_both(path_width, path_arc_toe)
    path_svg = get_path_svg(annot_id, rel_id, path_width, path_height, path_d, r_type, true)
    if (head_xcenter < tail_xcenter) {
      $head.append(path_svg)
    } else {
      $tail.append(path_svg)
    }
  } else if ((tail_ycenter - head_ycenter) > margin_error) {
    // Head is above Tail
    // Head
    path_width = tgt_extract_width - head_xcenter
    path_d = get_path_lefted(path_width, path_arc_toe)
    path_svg = get_path_svg(annot_id, rel_id, path_width, path_height, path_d, r_type, true)
    $head.append(path_svg)
    // Spare
    get_spare_svg(annot_id, rel_id, tgt_extract_width, path_height, path_arc_toe, r_type, $head, $tail)
    // Tail
    path_width = tail_xcenter
    path_d = get_path_righted(path_width, path_arc_toe)
    path_svg = get_path_svg(annot_id, rel_id, path_width, path_height, path_d, r_type, false)
    $tail.append(path_svg)
  } else {
    // Tail is above Head
    // Head
    path_width = head_xcenter
    path_d = get_path_righted(path_width, path_arc_toe)
    path_svg = get_path_svg(annot_id, rel_id, path_width, path_height, path_d, r_type, false)
    $head.append(path_svg)
    // Spare
    get_spare_svg(annot_id, rel_id, tgt_extract_width, path_height, path_arc_toe, r_type, $tail, $head)
    // Tail
    path_width = tgt_extract_width - tail_xcenter
    path_d = get_path_lefted(path_width, path_arc_toe)
    path_svg = get_path_svg(annot_id, rel_id, path_width, path_height, path_d, r_type, true)
    $tail.append(path_svg)
  }

  // Absolute Center
  set_svg_left(tgt_extract_width)
}

function get_path_svg(annot_id, rel_id, width, height, d, type, is_left) {
  var flag_left = is_left ? 'left' : 'right'
  var str = '<svg width="' + width + 'px" height="' + height + 'px" \
    xmlns="http://www.w3.org/2000/svg" version="1.1" \
    class="target-path ' + flag_left + ' target-path-' + rel_id + '" \
    id="' + annot_id + '_path_' + flag_left + '_' + rel_id + '"><path \
    d="' + d + '" stroke="' + get_rtype_color(type) + '" \
    stroke-width="2" stroke-dasharray="3 2" fill="none"/><text \
    fill="' + get_rtype_color(type) + '" font-family="Consolas" \
    font-size="12px" y="' + (height - 4) + 'px" \
    x="0px">' + get_verbosed_rtype(type) + '</text></svg>'

  return str
}

function get_spare_svg(annot_id, rel_id, width, height, arc_toe, type, $above, $below) {
  var $cur_head = $above
  var below_ycenter = get_ycenter($below)

  var path_d = null
  var path_svg = null

  var margin_error = 1e-1

  var cur_cnt = 0
  var max_cnt = 400

  // console.log('HEAD', rel_id, $cur_head.html())
  while (Math.abs(get_ycenter($cur_head) - below_ycenter) > margin_error && cur_cnt < max_cnt) {
    while (Math.abs(get_ycenter($cur_head.next()) - get_ycenter($cur_head)) <= margin_error) {
      // console.log('LEAP', rel_id, $cur_head.html())
      $cur_head = $cur_head.next()
    }
    $cur_head = $cur_head.next()
    // console.log('FINAL', rel_id, $cur_head.html())

    if ($cur_head[0].getClientRects().length > 1) {
      $cur_head.after('<span class="target-other"></span>')
      $cur_head = $cur_head.next()
    }

    if (Math.abs(get_ycenter($cur_head) - below_ycenter) <= margin_error) {
      break
    }

    path_d = get_path_none(width, arc_toe)
    path_svg = get_path_svg(annot_id, rel_id, width, height, path_d, type, true)
    $cur_head.append(path_svg)

    cur_cnt++
  }
}

function get_xcenter(obj) {
  return obj.position().left + obj.width() / 2
}

function get_ycenter(obj) {
  return obj.position().top + obj.height() / 2
}

function get_xbottom(obj) {
  return obj.position().left + obj.width()
}

function get_ybottom(obj) {
  return obj.position().top + obj.height()
}

function get_path_both(width, arc_toe) {
  var str = 'M 0 0 v 4 q 2 ' + arc_toe + ' 12 ' + arc_toe + ' H ' + (width - 12) + ' q 10 0 12 ' + (arc_toe * -1) + ' V 0'
  return str
}

function get_path_lefted(width, arc_toe) {
  var str = 'M 0 0 v 4 q 2 ' + arc_toe + ' 12 ' + arc_toe + ' H ' + width
  return str
}

function get_path_righted(width, arc_toe) {
  var str = 'M ' + width + ' 0 v 4 q -2 ' + arc_toe + ' -12 ' + arc_toe + ' H 0'
  return str
}

function get_path_none(width, arc_toe) {
  var str = 'M 0 ' + (arc_toe + 4) + ' H ' + width
  return str
}

function set_svg_left(tgt_width) {
  $(".target-path").each(function() {
    // SVG
    var tgt_path_width = $(this).width()

    var margin_error = 1e-1
    if (Math.abs(tgt_width - tgt_path_width) <= margin_error) {
      $(this).css('left', 'calc(0% - ' + $(this).parent().position().left + 'px')
    }

    // Text
    var $tgt_svg_text = $(this).children('text')
    var tgt_text_width = $tgt_svg_text[0].getBBox().width

    var tgt_path_left = $(this).parent().position().left + $(this).parent().width() / 2
    var tgt_text_left = (tgt_path_width > tgt_text_width) ? (tgt_path_width - tgt_text_width) / 2 : 0

    $tgt_svg_text.attr('x', '' + tgt_text_left + 'px')

    if (tgt_path_left + tgt_text_width > tgt_width) {
      $tgt_svg_text.hide()
    }

    if ((tgt_path_left < tgt_text_width) && $(this).hasClass('right')) {
      $tgt_svg_text.hide()
    }
  })
}

function token_mouseentered(obj) {
  $(obj).children('.target-badge').addClass('hovered')
  $(obj).children('.target-entity').addClass('hovered')
  $(obj).children('.target-caret').addClass('hovered')

  // var str = $(obj).attr('id').slice(-3)
  // console.log(str)
  // $('.target-path-' + str).addClass('hovered')
}

function token_mouseleaved(obj) {
  $(obj).children('.target-badge').removeClass('hovered')
  $(obj).children('.target-entity').removeClass('hovered')
  $(obj).children('.target-caret').removeClass('hovered')

  // var str = $(obj).attr('id').slice(-3)
  // console.log(str)
  // $('.target-path-' + str).removeClass('hovered')
}

function get_shorted_etype(verbose) {
  var verbose_to_short = {
    'Task': 'Task',
    'Method': 'Method',
    'Material': 'Material',
    'Metric': 'Metric',
    'Generic': 'Generic',
    'OtherScientificTerm': 'OST',
  }
  return verbose_to_short[verbose]
}

function get_verbosed_etype(short) {
  var short_to_verbose = {
    'Task': 'Task',
    'Method': 'Method',
    'Material': 'Material',
    'Metric': 'Metric',
    'Generic': 'Generic',
    'OtherScientificTerm': 'Other Scientific Term',
  }
  return short_to_verbose[short]
}

function get_shorted_rtype(verbose) {
  var verbose_to_short = {
    'Used-for': 'Used',
    'Feature-of': 'Feature',
    'Hyponym-of': 'Hyponym',
    'Evaluate-for': 'Evaluate',
    'Part-of': 'Part',
    'Compare': 'Compare',
    'Conjunction': 'Conjunction',
  }
  return verbose_to_short[verbose]
}

function get_verbosed_rtype(short) {
  var short_to_verbose = {
    'Used-for': 'Used for',
    'Feature-of': 'Feature of',
    'Hyponym-of': 'Hyponym of',
    'Evaluate-for': 'Evaluate for',
    'Part-of': 'Part of',
    'Compare': 'Compare',
    'Conjunction': 'Conjunction',
  }
  return short_to_verbose[short]
}

function get_rtype_color(verbose) {
  var verbose_to_color = {
    'Used-for': rtype_used_color,
    'Feature-of': rtype_feature_color,
    'Hyponym-of': rtype_hyponym_color,
    'Evaluate-for': rtype_evaluate_color,
    'Part-of': rtype_part_color,
    'Compare': rtype_compare_color,
    'Conjunction': rtype_conjunction_color,
  }
  return verbose_to_color[verbose]
}
