from django.shortcuts import redirect, render
from django.http import JsonResponse
import requests
import traceback


def page_extract(request):
    context = session_contrast(request)
    return render(request, './extract.html', context)


def page_sana(request):
    context = session_contrast(request)
    return render(request, './sana.html', context)


def page_text_classification_ch(request):
    context = session_contrast(request)
    return render(request, './text_classification_ch.html', context)


def page_translation(request):
    context = session_contrast(request)
    return render(request, './translation.html', context)


def page_mrc(request):
    context = session_contrast(request)
    return render(request, './mrc.html', context)


# 复制该函数，粘贴在该函数之上，并将 template 字段进行重命名，就像 page_extract 一样。
def page_template(request):
    context = session_contrast(request)
    return render(request, './template.html', context)


def page_not_found(request, exception, template_name=''):
    return redirect(page_extract)


def query_contrast(request):
    if request.is_ajax() and request.method == 'POST':
        # Fetch Source
        source_contrast = request.POST.get('source[contrast]', False)
        source_navbar = request.POST.get('source[navbar]', False)
        source_checkbox = request.POST.get('source[checkbox]', False)

        if not source_contrast or not source_navbar or not source_checkbox:
            return JsonResponse({'jcontrast': '__ERROR__'})

        request.session['contrast'] = source_contrast
        request.session['navbar'] = source_navbar
        request.session['checkbox'] = source_checkbox

        return JsonResponse({'jcontrast': '__SUCCESS__'})


def query_extract(request):
    if request.is_ajax() and request.method == 'POST':
        # Fetch Source
        source = request.POST.get('source', False)

        if not source:
            return JsonResponse({'jextract': '__ERROR__'})

        # Query
        try:
            jresponse = requests.post('http://localhost:2334/query_extarct',
                                      data={'source': source})
            jextract = jresponse.json()['jextract']
        except Exception:
            jextract = '__ERROR__'
            traceback.print_exc()

        return JsonResponse({'jextract': jextract})
    return render(request, './extract.html')


def query_text_classification_ch(request):
    if request.is_ajax() and request.method == 'POST':
        # Fetch Source
        source = request.POST.get('source', False)

        if not source:
            return JsonResponse({'jtext_classification_ch': '__ERROR__'})

        # Query
        try:
            jresponse = requests.post('http://localhost:2336/query_server',
                                      data={'source': source})
            jtext_classification_ch = jresponse.json()['jserver']
        except Exception:
            jtext_classification_ch = '__ERROR__'
            traceback.print_exc()

        return JsonResponse(
            {'jtext_classification_ch': jtext_classification_ch})
    return render(request, './text_classification_ch.html')


def query_translation(request):
    if request.is_ajax() and request.method == 'POST':
        # Fetch Source
        source = request.POST.get('source', False)

        if not source:
            return JsonResponse({'jtranslation': '__ERROR__'})

        # Query
        try:
            jresponse = requests.post(
                'http://localhost:2337/query_translation',
                data={'source': source})
            jtranslation = jresponse.json()['jserver']
        except Exception:
            jtranslation = '__ERROR__'
            traceback.print_exc()

        return JsonResponse({'jtranslation': jtranslation})
    return render(request, './translation.html')


def query_mrc(request):
    if request.is_ajax() and request.method == 'POST':
        # Fetch Source
        source = request.POST.get('source', False)

        if not source:
            return JsonResponse({'jmrc': '__ERROR__'})

        # Query
        try:
            jresponse = requests.post('http://localhost:2338/mrc',
                                      data={'source': source})
            jmrc = jresponse.json()['jmrc']
        except Exception:
            jmrc = '__ERROR__'
            traceback.print_exc()
        print(jmrc)
        return JsonResponse({'jmrc': jmrc})
    return render(request, './mrc.html')


def query_sana(request):
    if request.is_ajax() and request.method == 'POST':
        # Fetch Source
        source = request.POST.get('source', False)

        if not source:
            return JsonResponse({'jsana': '__ERROR__'})

        # Query
        try:
            jresponse = requests.post('http://localhost:2339/query_server',
                                      data={'source': source})
            jsana = jresponse.json()['jserver']
        except Exception:
            jsana = '__ERROR__'
            traceback.print_exc()

        return JsonResponse({'jsana': jsana})
    return render(request, './sana.html')


# 复制该函数，粘贴在该函数之上，并将 template 字段进行重命名，就像 query_extract 一样。
def query_template(request):
    if request.is_ajax() and request.method == 'POST':
        # Fetch Source
        source = request.POST.get('source', False)

        if not source:
            return JsonResponse({'jtemplate': '__ERROR__'})

        # Query
        try:
            jresponse = requests.post('http://localhost:2345/query_server',
                                      data={'source': source})
            jtemplate = jresponse.json()['jserver']
        except Exception:
            jtemplate = '__ERROR__'
            traceback.print_exc()

        return JsonResponse({'jtemplate': jtemplate})
    return render(request, './template.html')


def session_contrast(request):
    # Init
    if 'contrast' not in request.session:
        request.session['contrast'] = 'moon'
    if 'navbar' not in request.session:
        request.session['navbar'] = 'dark'
    if 'checkbox' not in request.session:
        request.session['checkbox'] = 'checked'
    # Refresh
    context = {}
    context['contrast'] = request.session.get('contrast')
    context['navbar'] = request.session.get('navbar')
    context['checkbox'] = request.session.get('checkbox')
    return context
