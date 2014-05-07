# -*- mode: python -*-
from kivy.tools.packaging.pyinstaller_hooks import install_hooks
install_hooks(globals())
a = Analysis(['main2.py'],
             pathex=['D:\\ISEF2014\\Programm\\8.0'],
             hiddenimports=[],
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='main2.exe',
          debug=True,
          strip=None,
          upx=True,
          console=True )
coll = COLLECT(exe, Tree('D:\\ISEF2014\\Programm\\8.0'),
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='main2')
