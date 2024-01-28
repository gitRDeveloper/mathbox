from django.db import models

# Create your models here.

class Chapter(models.Model):
    name = models.CharField(max_length=100)

class Formula(models.Model):
    chapter = models.ForeignKey(Chapter, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    formula_text = models.CharField(max_length=200)

class CalculationResult(models.Model):
    formula = models.ForeignKey(Formula, on_delete=models.CASCADE)
    input_values = models.JSONField()
    result = models.JSONField(null=True, blank=True)

class Topic(models.Model):
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name

class Question(models.Model):
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, null=True)
    question_text = models.CharField(max_length=200)
    choice_1 = models.CharField(max_length=200)
    choice_2 = models.CharField(max_length=200)
    choice_3 = models.CharField(max_length=200)
    choice_4 = models.CharField(max_length=200)
    correct_choice = models.PositiveSmallIntegerField(choices=[(1, 'Choice 1'), (2, 'Choice 2'), (3, 'Choice 3'), (4, 'Choice 4')])

    def __str__(self):
        return self.question_text