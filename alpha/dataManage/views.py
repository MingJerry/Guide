from django.shortcuts import render


# Create your views here.
def alpha_data_manage(request):
    return render(request, 'alpha-data-manage.html')


def alpha_data_console(request):
    return render(request, 'alpha-data-console.html')


def alpha_demo(request):
    return render(request, 'alpha-demo.html')
