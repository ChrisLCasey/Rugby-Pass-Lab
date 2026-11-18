# Pass Lab

A phone web app that measures a rugby pass from the phone's motion sensors and replays it with an England rugby ball.

Hold the phone like a ball, pass it (keeping hold), and it reports ball speed, launch angle, direction, spin, peak force, hang time and distance. A 3D England ball follows your phone live, then replays the pass over pitch markings.

There is no build step and no dependencies. It is plain HTML, CSS and JavaScript modules, with a small WebGL renderer for the ball.

## Deploy

1. Create a GitHub repo and push this folder to it.
   ```
   git init && git add . && git commit -m "Pass Lab"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
2. In Vercel choose **Add New, Project**, import the repo, and leave the framework preset as **Other**. Leave the build command and output directory empty. Deploy.
3. Open the Vercel URL on your phone. Motion sensors only work over HTTPS, which Vercel provides.

Test on a real phone. A desktop browser has no motion sensors, so there you can only use **Demo pass**, which runs the full analysis and replay on simulated data.

## Using it

1. Tap **Start pass**. On iPhone, allow motion access when asked.
2. During the 3-2-1 beeps, get into your passing stance and keep the phone still. Hold it as you would hold the ball, with the top edge of the phone as the nose of the ball, pointing at your target.
3. At the long beep ("Pass now") make your pass and finish with a firm stop. The app triggers on its own.
4. The ball replays your pass at half speed, then the numbers appear.

Safety: simulate the pass and keep a firm grip. Use a wrist strap if you can, and clear the space around you.

## What is measured, and how

| Result | How it is worked out |
| --- | --- |
| Ball speed | Gravity-free acceleration, rotated into world coordinates with the phone's orientation, integrated from the still start to the moment of peak hand speed (release). The phone is still at both ends, so leftover velocity is treated as sensor drift and removed. |
| Launch angle | Elevation of the velocity vector at release. |
| Direction | Horizontal angle of the release velocity relative to where the phone's nose pointed in your stance. |
| Spin and spiral | Gyroscope rate about the ball axis over the last 80 ms before release. Spiral is the share of all rotation that is about that axis. |
| Hang time, distance, highest point | Projectile maths from the release velocity and the release height in Settings. Air drag is ignored. |

## Accuracy: what to expect

Treat results as good for comparing one pass with another, not as lab measurements.

- Phone accelerometers often clip near 8 g, which can make very hard passes read low. The app warns when this happens.
- Phone gyroscopes usually top out around 2000 degrees per second, about 330 rpm. A real spin pass can spin faster than that, so very high spin reads low. The app warns about this too.
- Browsers deliver sensor data at roughly 60 readings a second. The analysis is tested at 100, 60 and 30 Hz.
- iPhone and Android differ in accelerometer sign conventions. The app checks this during the still countdown and corrects for it.
- A pass without a firm stop at the end cannot have its drift removed, so the speed is less certain. The app says so.

## Settings

Speed units (km/h, mph, m/s), which phone axis is the ball's long axis, release height, and trigger sensitivity. Settings and the last 12 passes are kept in the browser's local storage on that phone only.

## Files

```
index.html          page
css/styles.css      styles
js/main.js          app: state machine, replay, results, history, CSV export
js/analysis.js      pass analysis (pure functions, unit tested)
js/sensors.js       permission, sensor capture
js/viewer.js        WebGL2 ball renderer + pitch, trail
js/math.js          vector / quaternion / matrix helpers
js/demo.js          synthetic pass generator (Demo button and tests)
assets/ball.glb     England ball, 25k triangles, 1 MB
assets/ball_normal_os.jpg  baked surface detail
tools/              converts a new USDZ ball into the two assets above
tests/              unit tests
vercel.json         security and cache headers
```

Run the unit tests with `npm test` (Node 20 or newer, nothing to install).

## Changing the ball model

The original USDZ had about 423,000 triangles and a 20 MB size, too heavy for a phone page. `tools/bake.py` rebuilds it as a clean 25,000-triangle mesh and bakes the colours and surface detail into new textures.

```
unzip YourBall.usdz -d /tmp/ball_usdz
python3 tools/bake.py /tmp/ball_usdz assets
```

It needs Python with numpy, scipy and Pillow, and assumes the ball's long axis is the model's X axis, as Meshy exports it.

## Before sharing outside your team

The ball's texture carries a Gilbert-style wordmark and the England rose, as generated in the original model. Check you have the right to use them before publishing the app publicly or in client material.
