const gcStyle = getComputedStyle(document.body)
const docStyle = document.documentElement.style;

// Entity
const etype_task_color = gcStyle.getPropertyValue('--entity-type-task-color')
const etype_method_color = gcStyle.getPropertyValue('--entity-type-method-color')
const etype_metric_color = gcStyle.getPropertyValue('--entity-type-metric-color')
const etype_material_color = gcStyle.getPropertyValue('--entity-type-material-color')
const etype_generic_color = gcStyle.getPropertyValue('--entity-type-generic-color')
const etype_otherscientificterm_color = gcStyle.getPropertyValue('--entity-type-otherscientificterm-color')

// Relation
const rtype_used_color = gcStyle.getPropertyValue('--relation-type-used-color')
const rtype_feature_color = gcStyle.getPropertyValue('--relation-type-feature-color')
const rtype_hyponym_color = gcStyle.getPropertyValue('--relation-type-hyponym-color')
const rtype_evaluate_color = gcStyle.getPropertyValue('--relation-type-evaluate-color')
const rtype_part_color = gcStyle.getPropertyValue('--relation-type-part-color')
const rtype_compare_color = gcStyle.getPropertyValue('--relation-type-compare-color')
const rtype_conjunction_color = gcStyle.getPropertyValue('--relation-type-conjunction-color')

// Initialization
$(function() {
  // Contrast
  $('#switch_contrast').on('click', function() {
    toggle_scrollbar()
  })
  toggle_scrollbar()
})

function toggle_scrollbar() {
  docStyle.setProperty('--scroll-bar-color', gcStyle.getPropertyValue('--' + get_contrast() + '-stress-color'))
}
