from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import cgi, cgitb
import datetime
from rest_framework.decorators import list_route
from rest_framework import serializers, renderers
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework.views import APIView
from .models import QuestionDemo
import base64


# Create your views here.
def alpha_data_manage(request):
    return render(request, 'alpha-data-manage.html')


def alpha_data_console(request):
    return render(request, 'alpha-data-console.html')


def alpha_demo(request):
    return render(request, 'alpha-demo.html')


class AlphaDemoSerializer(serializers.HyperlinkedModelSerializer):
    Q_INDEX = serializers.IntegerField()
    QUESTION_USER = serializers.CharField(max_length=255)
    ANSWER_USER = serializers.CharField(max_length=255)
    QUEST_DATETIME = serializers.DateTimeField()
    PRE_METHOD = serializers.IntegerField()

    class Meta:
        model = QuestionDemo
        fields = ('Q_INDEX',
                  'QUESTION_USER',
                  'ANSWER_USER',
                  'QUEST_DATETIME',
                  'PRE_METHOD',
                  )


class AlphaDemoViewSet(APIView):

    @csrf_exempt
    @list_route(methods=['post'], renderer_classes=(renderers.JSONRenderer,))
    def save_and_run(self):
        # from prediction.preProcess import trigger
        #
        # trigger.pre_for_input(self.POST['zhengzhuang'])
        print("OK")
        return JsonResponse(data={"status": 1,
                                  "message": "Prediction Update Success",
                                  "clinicA": "OK"
                                  })

