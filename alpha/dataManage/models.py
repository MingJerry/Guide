from django.db import models


class QaList(models.Model):
    QAINDEX = models.IntegerField(primary_key=True)
    CLINIC = models.TextField()
    QUESTION = models.TextField()
    ANSWER = models.TextField()
    SNAPSHOT = models.DateField()

    def __str__(self):
        return self.CLINIC


class QuestionDemo(models.Model):
    Q_INDEX = models.AutoField(primary_key=True)
    QUESTION_USER = models.TextField()
    ANSWER_USER = models.TextField()
    QUEST_DATETIME = models.DateTimeField('date published')
    PRE_METHOD = models.IntegerField()

    def __str__(self):
        return self.QUESTION_USER


class QaAdmin(models.Model):
    Qa_INDEX = models.AutoField(primary_key=True)
    QUESTION_ADMIN = models.TextField()
    ANSWER_ADMIN = models.TextField()
    QA_DATETIME = models.DateTimeField('date published')

    def __str__(self):
        return self.QUESTION_ADMIN
