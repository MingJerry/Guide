from django.db import models


class TestAlpha(models.Model):
    name = models.CharField(max_length=20)
