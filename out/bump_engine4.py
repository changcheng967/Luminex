import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'OpenSite.settings')
django.setup()
from OpenBench.models import Engine, EngineConfig, Test, Result

NEWSHA = 'cd92c1716664f04ad1136fd408ff8f3b7e1133c1'
SRC = f'https://github.com/changcheng967/Luminex/archive/{NEWSHA}.zip'
Engine.objects.update_or_create(name='Luminex', defaults=dict(source=SRC, sha=NEWSHA, bench=206603))
EngineConfig.objects.filter(name='Luminex').update(source=SRC, build_compilers='g++-13 clang++')
print('Engine:', Engine.objects.get(name='Luminex').sha[:8],
      '| compilers:', EngineConfig.objects.get(name='Luminex').build_compilers,
      '| Tests:', Test.objects.count())
