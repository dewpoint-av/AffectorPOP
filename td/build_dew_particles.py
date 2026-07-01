# build_dew_particles.py  —  self-contained particle-sim COMP wrapping the Dew Affector
# =============================================================================
# Builds a "DewParticles" Base COMP that hides the feedback loop:
#
#     in1 -> Feedback -> Copy(1) -> Dew Affector(Integrate ON) -> Null -> out1
#                ^-- Feedback targetpop = Null (the loop)
#
# So the user drops ONE node, wires a point source into it, hits play, and the
# points accumulate motion — no manual feedback wiring.
#
# Baked-in freeze workarounds for the KNOWN TouchDesigner "POP Feedback" regression
# (Derivative-acknowledged; forum.derivative.ca/t/779867):
#   - a Copy POP (1 copy) immediately AFTER the Feedback POP,
#   - viewers turned OFF on every node inside the loop.
#
# ⚠️ REQUIRES the AffectorPOP.dll installed (Documents/Derivative/Plugins/). A .tox
#    can't embed a C++ DLL, so ship the DLL alongside the .tox and install it first.
#
# RUN: paste into a Textport / Script DAT and call build_dew_particles().
#      Re-running destroys+rebuilds the COMP. Report any errors (untested on your TD).
# =============================================================================


def _try_create(parent, type_or_name, name):
    """Create a node, tolerating custom-op vs built-in type differences across TD versions."""
    try:
        return parent.create(type_or_name, name)
    except Exception:
        # Custom operators are created by their optype string on some TD builds.
        try:
            return parent.create(str(type_or_name), name)
        except Exception as e:
            print('build_dew_particles: create failed for', type_or_name, '->', e)
            return None


def build_dew_particles(parent_path='/project1', name='DewParticles'):
    parent = op(parent_path)
    if parent is None:
        raise RuntimeError('parent_path %r not found' % parent_path)
    if parent.op(name):
        parent.op(name).destroy()

    comp = parent.create(baseCOMP, name)
    comp.color = (0.45, 0.7, 0.45)

    # ---- inner nodes ---------------------------------------------------------
    gin  = comp.create(inPOP,       'in1');        gin.nodeX, gin.nodeY   = -600, 0
    fb   = comp.create(feedbackPOP, 'feedback1');  fb.nodeX,  fb.nodeY    = -400, 0
    cp   = comp.create(copyPOP,     'copy1');      cp.nodeX,  cp.nodeY    = -250, 0   # freeze workaround
    null = comp.create(nullPOP,     'null1');      null.nodeX, null.nodeY = 100,  0
    gout = comp.create(outPOP,      'out1');       gout.nodeX, gout.nodeY = 250,  0

    # Dew Affector (installed custom op). If auto-create fails on this TD build, drop one in manually.
    aff = _try_create(comp, 'dewaffector', 'affector1')
    if aff is not None:
        aff.nodeX, aff.nodeY = -50, 0

    # ---- wiring: in1 -> feedback -> copy -> affector -> null -> out1 ----------
    fb.inputConnectors[0].connect(gin)
    cp.inputConnectors[0].connect(fb)
    if aff is not None:
        aff.inputConnectors[0].connect(cp)
        null.inputConnectors[0].connect(aff)
    else:
        null.inputConnectors[0].connect(cp)
        print('build_dew_particles: NOTE — no Dew Affector; wire one between copy1 and null1 manually.')
    gout.inputConnectors[0].connect(null)

    # ---- close the loop: Feedback targets the Null (param ref, NOT a wire cycle) --
    for tok in ('targetpop', 'Targetpop', 'target'):
        p = getattr(fb.par, tok, None)
        if p is not None:
            try: p.val = null.path
            except Exception:
                try: p.val = null
                except Exception: pass
            break

    # ---- Affector defaults for a real feedback loop --------------------------
    if aff is not None:
        for tok, val in (('Integrate', 1), ('Selfsim', 0)):
            p = getattr(aff.par, tok, None)
            if p is not None:
                try: p.val = val
                except Exception: pass

    # ---- freeze workaround: viewers OFF inside the loop ----------------------
    for o in (gin, fb, cp, null, gout, aff):
        if o is None:
            continue
        try: o.viewer = False
        except Exception: pass

    # ---- promote the Affector's custom params onto the COMP (bind) -----------
    if aff is not None:
        page = comp.appendCustomPage('Particles')
        try:
            for src in aff.customPars:
                if src.name in ('Selfsim', 'Reset'):   # loop-managed / internal-only
                    continue
                # mirror by style with a bind back to the affector par
                st = src.style
                dst = None
                if st == 'Float':   dst = page.appendFloat(src.name, label=src.label)[0]
                elif st == 'Int':   dst = page.appendInt(src.name, label=src.label)[0]
                elif st == 'Toggle':dst = page.appendToggle(src.name, label=src.label)[0]
                elif st == 'Menu':  dst = page.appendMenu(src.name, label=src.label)[0]
                elif st in ('XYZ', 'RGB', 'XY'):
                    fn = {'XYZ': page.appendXYZ, 'RGB': page.appendRGB, 'XY': page.appendXY}[st]
                    grp = fn(src.name, label=src.label)
                    for gi, gp in enumerate(grp):
                        try:
                            gp.bindExpr = "op('affector1').par.%s" % src.tupletName if False else ''
                        except Exception: pass
                    continue
                if dst is not None:
                    try:
                        dst.bindExpr = "op('affector1').par.%s" % src.name
                        dst.mode = ParMode.BIND
                    except Exception:
                        pass
        except Exception as e:
            print('build_dew_particles: param promotion skipped ->', e)

    print('build_dew_particles: built', comp.path,
          '(wire a point POP into its input; play the timeline).')
    print('  Requires AffectorPOP.dll installed. Copy POP + viewers-off are the freeze workarounds.')
    return comp


# Run on paste:
build_dew_particles()
