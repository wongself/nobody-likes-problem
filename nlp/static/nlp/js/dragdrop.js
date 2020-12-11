var $dragNdrop = $('.page-main')
var lastEnter = null

$(function() {
  $dragNdrop.on('dragenter', function(e) {
    // Prevent Default
    e.stopPropagation()
    e.preventDefault()

    // Drag Enter
    lastEnter = e.target

    // Print Drag & Drop
    // console.log('Dragenter ', lastEnter)

    $dragNdrop.addClass('dragging-over');
    $('#mask_extract_drag').fadeIn()
  })

  $dragNdrop.on('dragleave', function(e) {
    // Prevent Default
    e.stopPropagation()
    e.preventDefault()

    // Drag Leave
    if (lastEnter == e.target) {
      // Print Drag & Drop
      // console.log('Dragleave ', lastEnter)

      $dragNdrop.removeClass('dragging-over')
      $('#mask_extract_drag').fadeOut()
    }
  })

  $dragNdrop.on('dragover', function(e) {
    // Prevent Default
    e.stopPropagation()
    e.preventDefault()

    // Drop Effect
    e.originalEvent.dataTransfer.dropEffect = 'move';
  })

  $dragNdrop.on('drop', function(e) {
    // Prevent Default
    e.stopPropagation()
    e.preventDefault()

    // Print Input
    // console.log('Drop ', e)

    var files = e.originalEvent.dataTransfer.files
    if (files.length > 1) {
      raise_modal_error('只允许识别一个文件！')
      console.error('One file only.')
    } else if (!is_empty(files)) {
      upload_document(files[0])
    }

    $dragNdrop.removeClass('dragging-over')
    $('#mask_extract_drag').fadeOut()
  })
})
