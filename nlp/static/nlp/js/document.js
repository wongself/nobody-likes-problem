var max_file_size = 8 * 1024 * 1024

// Upload Document
function upload_document(file) {
  // Print File
  // console.log(file)

  // File Validate
  if (file.size <= 0) {
    raise_modal_error('文件无效！')
    return
  }

  if (file.size > max_file_size) {
    raise_modal_error('文件不能大于' + max_file_size / 1024 / 1024 + 'MB！')
    return
  }

  // File Extract
  var fname = file.name
  var ftype = (fname.substring(fname.lastIndexOf('.') + 1, fname.length)).toLowerCase()

  try {
    switch (ftype) {
      case 'txt':
        extract_text_from_txt(file)
        break
      case 'docx':
      case 'pptx':
        extract_text_from_docx(file)
        break
      default:
        raise_modal_error('文件格式暂不支持！')
        return
    }
  } catch (e) {
    raise_modal_error('未知错误，请重试！')
    console.error(e)
  }
}

// Read File
function extract_text_from_txt(file) {
  var reader = new FileReader()
  reader.onload = function() {
    var text = this.result
    extract_text_success(text)
  }
  reader.onerror = function(e) {
    extract_text_error(e)
  }
  reader.readAsText(file)
}

function extract_text_from_docx(file) {
  var reader = new FileReader()
  reader.onload = function() {
    var zip = new PizZip(reader.result)
    var doc = new window.docxtemplater(zip)
    var text = doc.getFullText()
    extract_text_success(text)
  }
  reader.onerror = function(e) {
    extract_text_error(e)
  }
  reader.readAsBinaryString(file)
}

// Extract Status
function extract_text_success(text) {
  // Source Textarea
  $('#left_text_area').val(text.substring(0, max_length))
  toggle_textarea_placeholder()
  // Query Trigger
  $('#extract_button').click()
}

function extract_text_error(e) {
  raise_modal_error('上传失败，请重试！')
  console.error(e)
}
