import json
from django.shortcuts import render


def trans_home(request):
    testlist = ['test1', 'test2']
    testdict = {'site': 'trans', 'author': 'XM'}
    return render(request, 'trans.html', {'List': json.dumps(testlist),
                                          'Dict': json.dumps(testdict)})
