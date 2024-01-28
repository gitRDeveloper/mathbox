from rest_framework import serializers
from .models import Chapter, Formula, CalculationResult

class ChapterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chapter
        fields = '__all__'

class FormulaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Formula
        fields = '__all__'

class CalculationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = CalculationResult
        fields = '__all__'

    def create(self, validated_data):
        # Custom logic to create CalculationResult objects
        result = validated_data.get('result', None)  # Get the 'result' field from validated_data
        # Perform custom processing here
        return CalculationResult.objects.create(**validated_data)

