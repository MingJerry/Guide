from django.db import models


class QaList(models.Model):
    QAINDEX = models.IntegerField(primary_key=True)
    CLINIC = models.TextField()
    QUESTION = models.TextField()
    ANSWER = models.TextField()
    SNAPSHOT = models.DateField()

    def __str__(self):
        return self.CLINIC
