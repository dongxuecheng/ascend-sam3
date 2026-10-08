// Run with Node 20+: node --test tests/frontend.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const ui = require('../service/static/script.js');

const file = new Blob(['test-image'], { type: 'image/png' });
const options = { mode: 'multi-class', labels: ['person', 'fire'], confidence: 0.3, returnMask: true,
    preLabels: ['person'], mergeResults: false, preConfidence: '', crop: { enable_ar_fix: true }, maxCrops: 4 };

test('normalizes labels without interpreting HTML', () => {
    assert.deepEqual(ui.normalizeLabels([' Person ', 'PERSON', '', '<img src=x>']), ['Person', '<img src=x>']);
});
test('ordinary inference preserves existing file protocol', () => {
    const { endpoint, formData } = ui.buildInferenceForm(file, options);
    assert.equal(endpoint, '/predict/file');
    assert.deepEqual(formData.getAll('class_names'), ['person', 'fire']);
    assert.equal(formData.get('return_mask'), 'true');
    assert.equal(formData.has('mode'), false);
    assert.equal(formData.has('crop_config_json'), false);
});
test('refinement uses the dedicated route and budgets', () => {
    const { endpoint, formData } = ui.buildInferenceForm(file, { ...options, mode: 'obj-refine',
        preConfidence: 0, crop: { padding: 0, w_diou: 0, max_crops: 2, enable_ar_fix: false } });
    assert.equal(endpoint, '/predict-obj-refine/file');
    assert.deepEqual(formData.getAll('pre_detect_labels'), ['person']);
    assert.equal(formData.get('merge_results'), 'false');
    assert.equal(formData.get('pre_detect_confidence'), '0');
    const cfg = JSON.parse(formData.get('crop_config_json'));
    assert.equal(cfg.padding, 0); assert.equal(cfg.w_diou, 0); assert.equal(cfg.max_crops, 2);
    assert.equal(cfg.enable_ar_fix, false);
});
test('blank crop budget lets server apply env defaults', () => {
    assert.equal(ui.getCropConfig({ max_crops: '', enable_ar_fix: true }).max_crops, undefined);
});
test('rejects unsupported geometry and invalid numeric settings', () => {
    for (const mode of ['box', 'mixed', 'from-image']) assert.throws(() => ui.buildInferenceForm(file, { ...options, mode }));
    for (const value of [0, 5, 1.5, 'NaN']) assert.throws(() => ui.getCropConfig({ max_crops: value }, 4));
    assert.throws(() => ui.getCropConfig({ padding: -1 }));
    assert.throws(() => ui.buildInferenceForm(file, { ...options, confidence: Infinity }));
    assert.throws(() => ui.buildInferenceForm(file, { ...options, labels: [] }));
});
test('filter/sort keep stable original indices and checked targets', () => {
    const results = [
        { label: 'helmet', score: 0.8, box: [0, 0, 20, 20] },
        { label: 'person', score: 0.7, box: [0, 0, 100, 100] },
        { label: 'helmet', score: 0.9, box: [0, 0, 40, 40] },
    ];
    const settings = { labels: new Set(['helmet']), confidence: 0, area: 0, checked: new Set([0]), onlyChecked: false, sort: 'score_desc' };
    assert.deepEqual(ui.visibleResults(results, settings).map(r => r.index), [2, 0]);
    assert.deepEqual(ui.visibleResults(results, { ...settings, onlyChecked: true }).map(r => r.index), [0]);
    assert.equal(ui.visibleResults(results, { ...settings, area: 2000 }).length, 0);
});
test('decodes bbox-local row-major 1-based mask RLE', () => {
    const pixels = ui.decodeRle([1, 1, 4, 1], 2, 2, [255, 0, 0]);
    assert.deepEqual([...pixels], [255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255]);
    assert.throws(() => ui.decodeRle([], 0, 2, [0, 0, 0]));
    assert.throws(() => ui.decodeRle([], 100000, 100000, [0, 0, 0]));
});
test('shows FastAPI validation detail and gateway failures', () => {
    assert.equal(ui.formatError({ detail: [{ msg: 'bad max_crops' }] }, 422), 'bad max_crops');
    assert.equal(ui.formatError({ detail: 'invalid image' }, 400), 'invalid image');
    assert.match(ui.formatError(null, 502), /502/);
});
test('every JS DOM reference exists in HTML, geo modes stay disabled', () => {
    const script = fs.readFileSync('service/static/script.js', 'utf8');
    const html = fs.readFileSync('service/static/index.html', 'utf8');
    for (const [, id] of script.matchAll(/\$\('([^']+)'\)/g)) assert.ok(html.includes(`id="${id}"`), id);
    for (const mode of ['box', 'mixed', 'from-image']) assert.ok(html.includes(`value="${mode}" disabled`));
    assert.ok(html.includes('/static/style.css?v='));
    assert.ok(html.includes('/static/script.js?v='));
});
