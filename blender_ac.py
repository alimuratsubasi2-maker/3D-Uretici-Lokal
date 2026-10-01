import sys

import bpy

stl_yolu = sys.argv[sys.argv.index("--") + 1]


def yukle():
    varsayilan_kup = bpy.data.objects.get("Cube")
    if varsayilan_kup is not None:
        bpy.data.objects.remove(varsayilan_kup, do_unlink=True)

    bpy.ops.wm.stl_import(filepath=stl_yolu)

    # Arayuz tam yuklenmeden view3d operatorleri calismaz; bu yuzden timer icinden
    # 3D gorunum alanini bulup kamerayi modele odakla.
    for pencere in bpy.context.window_manager.windows:
        for alan in pencere.screen.areas:
            if alan.type == "VIEW_3D":
                bolge = next(r for r in alan.regions if r.type == "WINDOW")
                with bpy.context.temp_override(window=pencere, area=alan, region=bolge):
                    bpy.ops.view3d.view_selected()
    return None


bpy.app.timers.register(yukle, first_interval=0.5)
