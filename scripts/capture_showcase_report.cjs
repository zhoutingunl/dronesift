#!/usr/bin/env node
/* Capture the real showcase report and exercise its linked controls.
 * Build-only dependency: Playwright; no Node packages are needed by dronesift.
 * See docs/showcase.md for installation and reproduction commands.
 */
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {pathToFileURL} = require('node:url');

function option(name, fallback) {
  const i = process.argv.indexOf(name);
  if (i === -1) return fallback;
  if (!process.argv[i + 1] || process.argv[i + 1].startsWith('--')) {
    throw new Error(`Missing value for ${name}`);
  }
  return process.argv[i + 1];
}

async function main() {
  if (process.argv.includes('--help')) {
    console.log('node scripts/capture_showcase_report.cjs [--report FILE] [--output-dir DIR] ' +
      '[--playwright-module PATH] [--browser-channel chrome] [--browser-executable PATH]');
    return;
  }
  const modulePath = option('--playwright-module');
  const {chromium} = require(modulePath ? path.resolve(modulePath) : 'playwright');
  const report = path.resolve(option('--report', 'docs/examples/flight-report.html'));
  const output = path.resolve(option('--output-dir', 'docs/media'));
  const manifestPath = path.join(output, 'analysis-sources.json');
  if (!fs.existsSync(report) || !fs.existsSync(manifestPath)) {
    throw new Error('Build the report, clip and manifest with build_showcase_report.py first.');
  }
  const launch = {headless: true, args: [
    '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  ]};
  const channel = option('--browser-channel');
  const executablePath = option('--browser-executable');
  if (channel) launch.channel = channel;
  if (executablePath) launch.executablePath = executablePath;
  const browser = await chromium.launch(launch);
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 900},
      deviceScaleFactor: 1, colorScheme: 'light', ignoreHTTPSErrors: true});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(report).href, {waitUntil: 'load', timeout: 60000});
    await page.waitForFunction(() =>
      document.querySelector('#p3d')?.data?.some(t => t.type === 'scatter3d') &&
      document.querySelector('#p2d')?.data?.length > 0, null, {timeout: 30000});
    const metadata = await page.evaluate(() => ({
      source: M.name, segments: M.segs.length, duration_s: M.duration,
      timeline_end_s: T_END,
      thumbnail_count: M.thumbs?.count,
    }));
    if (metadata.segments !== 3 || !metadata.thumbnail_count) {
      throw new Error('Expected the reviewed three-segment report and embedded thumbnails.');
    }
    const settle = async () => {
      await page.evaluate(() => new Promise(resolve =>
        requestAnimationFrame(() => requestAnimationFrame(resolve))));
      await page.waitForTimeout(350);
    };

    await page.locator('.chip').nth(1).click();
    await settle();
    const chipWorks = await page.evaluate(() =>
      Math.abs(range[0] - M.segs[1].start) < 0.001 &&
      Math.abs(document.querySelector('#p2d')._fullLayout.xaxis.range[1] - M.segs[1].end) < 0.001);
    await page.locator('#t0').fill('16');
    await page.locator('#t1').fill('40');
    await page.locator('#t1').dispatchEvent('change');
    await settle();
    const inputsWork = await page.evaluate(() => range[0] === 16 && range[1] === 40);
    await page.evaluate(() => Plotly.relayout('p2d', {'xaxis.range[0]': 18, 'xaxis.range[1]': 38}));
    await settle();
    const plotZoomWorks = await page.evaluate(() => range[0] === 18 && range[1] === 38);
    await page.locator('.th').nth(1).click();
    const thumbnailWorks = await page.evaluate(() => !document.querySelector('#preview').hidden &&
      document.querySelector('#preview .im').style.backgroundImage.includes('data:image/jpeg;base64,'));
    await page.locator('#reset').click();
    await settle();
    const resetWorks = await page.evaluate(() => range[0] === 0 && range[1] === T_END);
    const video = page.locator('video');
    if (await video.count() !== 1) throw new Error('The showcase needs its matching selected video.');
    await video.evaluate(el => el.load());
    await page.waitForFunction(() => document.querySelector('video')?.readyState >= 1);
    const videoInfo = await video.evaluate(el => ({duration_s: el.duration,
      width: el.videoWidth, height: el.videoHeight,
      source: el.getAttribute('src') || el.querySelector('source')?.getAttribute('src')}));
    if (Math.abs(videoInfo.duration_s - 8) > 0.001 || videoInfo.width !== 960 || videoInfo.height !== 540) {
      throw new Error('Selected-video metadata differs from the documented 8-second preview.');
    }
    if (errors.length || ![chipWorks, inputsWork, plotZoomWorks, thumbnailWorks, resetWorks].every(Boolean)) {
      throw new Error(`Report interaction failed: ${JSON.stringify({errors, chipWorks,
        inputsWork, plotZoomWorks, thumbnailWorks, resetWorks})}`);
    }

    const cameraEye = {x: 1.7, y: -2.4, z: 1.7};
    await page.evaluate(eye => Plotly.relayout('p3d', {'scene.camera.eye': eye}), cameraEye);
    await page.evaluate(() => { document.querySelector('#preview').hidden = true; });
    await settle();
    const overview = path.join(output, 'analysis-report.png');
    await page.locator('.wrap').screenshot({path: overview, animations: 'disabled'});
    await page.evaluate(() => { setRange(16, 40); showPreview(20); });
    await settle();
    const detail = path.join(output, 'analysis-timeline.png');
    await page.locator('.p2dwrap').screenshot({path: detail, animations: 'disabled'});

    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const captures = [overview, detail].map(filename => {
      const buffer = fs.readFileSync(filename);
      return {file: path.basename(filename), bytes: buffer.length,
        sha256: crypto.createHash('sha256').update(buffer).digest('hex'),
        width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20)};
    });
    manifest.browser_capture = {generator: 'scripts/capture_showcase_report.cjs',
      browser: browser.version(), viewport: {width: 1440, height: 900}, device_scale_factor: 1,
      camera_eye: cameraEye,
      color_scheme: 'light', overview_range_s: [0, metadata.timeline_end_s], detail_range_s: [16, 40],
      hovered_preview_source_s: 20, relative_report: '../examples/flight-report.html',
      ui_checks: {chipWorks, inputsWork, plotZoomWorks, thumbnailWorks, resetWorks},
      selected_video: videoInfo, javascript_errors: errors, screenshots: captures};
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
    console.log(JSON.stringify({verified: true, screenshots: captures,
      interactive_checks: manifest.browser_capture.ui_checks, selected_video: videoInfo}, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
