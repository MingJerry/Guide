from django.shortcuts import render


# Create your views here.
def alpha_data_manage(request):
    return render(request, 'alpha-data-manage.html')
