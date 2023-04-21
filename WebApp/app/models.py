from django.db import models
from django.contrib.auth.models import User

class UserData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    translations_count = models.PositiveIntegerField()
    feedback_count = models.PositiveIntegerField()

class UserFeedback(models.Model):
    src_text = models.TextField()
    target_text = models.TextField()
    lang_pair = models.CharField(max_length=10)