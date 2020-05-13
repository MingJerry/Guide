from django.contrib import admin
from .models import QaList, QuestionDemo, QaAdmin


class QaListAdmin(admin.ModelAdmin):
    list_display = ('QAINDEX', 'CLINIC', 'QUESTION')
    list_filter = ['CLINIC']
    search_fields = ['QUESTION']
    list_per_page = 15


admin.site.register(QaList, QaListAdmin)
admin.site.register(QuestionDemo)
admin.site.register(QaAdmin)

