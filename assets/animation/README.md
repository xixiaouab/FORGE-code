# FORGE Animation Source

`motion.html` contains the project page’s six-scene walkthrough. It uses the bundled Comic font; its license is in `assets/motion-font-OFL.txt`.

The README GIF is a 36-second loop exported from this walkthrough. Browser frames are cropped above the playback controls so the illustration and subtitles remain visible.

To assemble a new export from numbered browser screenshots with Pillow:

```bash
python render_gif.py frames ../forge-overview.gif --timestamps capture-times.json
```

`capture-times.json` records the timing of the current export. The PNG frames are not included in the repository.
