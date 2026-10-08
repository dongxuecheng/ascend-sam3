/* Ascend SAM3 UI: upstream dark layout, native Ascend API adapter.
 * All label text is inserted as text, not HTML. No CDN or frontend build needed.
 */
(function () {
    'use strict';

    function normalizeLabels(values) {
        const seen = new Set();
        return values.map(value => String(value).trim()).filter(value => {
            if (!value || seen.has(value.toLowerCase())) return false;
            seen.add(value.toLowerCase());
            return true;
        });
    }

    const cropFields = {
        max_size: [640, 1, 16384, true], padding: [20, 0, 4096, true],
        w_diou: [30, 0, 100000], w_expansion: [5, 0, 100000],
        count_penalty: [120, 0, 100000], nms_threshold: [0.2, 0, 1],
        target_ar: [1, 0.1, 10],
    };

    function numeric(value, name, min, max, integer = false) {
        const number = Number(value);
        if (!Number.isFinite(number) || number < min || number > max ||
            (integer && !Number.isInteger(number))) {
            throw new Error(`${name} 必须在 ${min}～${max} 之间${integer ? '，且为整数' : ''}`);
        }
        return number;
    }

    function getCropConfig(values, maxCrops = 32) {
        const cfg = {};
        for (const [name, [fallback, min, max, integer]] of Object.entries(cropFields)) {
            const value = values[name];
            cfg[name] = numeric(value === '' || value == null ? fallback : value, name, min, max, integer);
        }
        cfg.enable_ar_fix = Boolean(values.enable_ar_fix);
        if (values.max_crops !== '' && values.max_crops != null) {
            cfg.max_crops = numeric(values.max_crops, 'max_crops', 1, maxCrops, true);
        }
        return cfg;
    }

    function buildInferenceForm(file, options) {
        const mode = options.mode;
        if (!['multi-class', 'obj-refine'].includes(mode)) {
            throw new Error('当前昇腾后端不支持这个推理模式');
        }
        const labels = normalizeLabels(options.labels);
        if (!labels.length) throw new Error('请至少保留一个识别标签');
        if (labels.length > 32 || labels.some(label => label.length > 256)) {
            throw new Error('最多支持 32 个标签，每个标签不超过 256 字符');
        }
        const formData = new FormData();
        formData.append('image', file);
        labels.forEach(label => formData.append('class_names', label));
        formData.append('confidence', String(numeric(options.confidence, '置信度', 0, 1)));
        formData.append('return_mask', String(Boolean(options.returnMask)));
        if (mode === 'obj-refine') {
            const preLabels = normalizeLabels(options.preLabels);
            if (!preLabels.length) throw new Error('请至少保留一个预检测标签');
            if (preLabels.length > 32 || preLabels.some(label => label.length > 256)) {
                throw new Error('最多支持 32 个预检测标签，每个标签不超过 256 字符');
            }
            preLabels.forEach(label => formData.append('pre_detect_labels', label));
            formData.append('merge_results', String(Boolean(options.mergeResults)));
            formData.append('crop_config_json', JSON.stringify(getCropConfig(options.crop, options.maxCrops)));
            if (options.preConfidence !== '' && options.preConfidence != null) {
                formData.append('pre_detect_confidence', String(numeric(options.preConfidence, '预检测阈值', 0, 1)));
            }
        }
        return { endpoint: mode === 'obj-refine' ? '/predict-obj-refine/file' : '/predict/file', formData };
    }

    function formatError(data, status) {
        const detail = data && (data.error || data.detail);
        if (typeof detail === 'string') return detail;
        if (Array.isArray(detail)) return detail.map(item => item.msg || '参数不合法').join('；');
        return `请求失败（HTTP ${status}）`;
    }

    function resultArea(item) {
        if (!Array.isArray(item.box) || item.box.length !== 4) return 0;
        return Math.max(0, item.box[2] - item.box[0]) * Math.max(0, item.box[3] - item.box[1]);
    }

    function visibleResults(results, settings) {
        const items = results.map((item, index) => ({ item, index })).filter(({ item, index }) =>
            settings.labels.has(item.label) && item.score >= settings.confidence &&
            resultArea(item) >= settings.area && (!settings.onlyChecked || settings.checked.has(index)));
        const sorts = {
            area_desc: (a, b) => resultArea(b.item) - resultArea(a.item),
            area_asc: (a, b) => resultArea(a.item) - resultArea(b.item),
            score_desc: (a, b) => b.item.score - a.item.score,
            score_asc: (a, b) => a.item.score - b.item.score,
        };
        if (sorts[settings.sort]) items.sort(sorts[settings.sort]);
        return items;
    }

    function decodeRle(rle, width, height, rgb) {
        if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1 ||
            width * height > 50000000) throw new Error('Mask 尺寸无效或过大');
        const pixels = new Uint8ClampedArray(width * height * 4);
        for (let i = 0; i + 1 < rle.length; i += 2) {
            const start = rle[i] - 1, count = rle[i + 1];
            if (!Number.isSafeInteger(start) || !Number.isSafeInteger(count) || count <= 0) continue;
            const end = Math.min(width * height, start + count);
            for (let index = Math.max(0, start); index < end; index++) {
                const p = index * 4;
                pixels[p] = rgb[0]; pixels[p + 1] = rgb[1]; pixels[p + 2] = rgb[2]; pixels[p + 3] = 255;
            }
        }
        return pixels;
    }

    // The request/filter/RLE helpers can be tested with Node without a browser.
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = { normalizeLabels, getCropConfig, buildInferenceForm, formatError, resultArea, visibleResults, decodeRle };
    }
    if (typeof document === 'undefined') return;

    const $ = id => document.getElementById(id);
    const state = {
        file: null, image: null, imageUrl: null, loadVersion: 0, requestVersion: 0,
        results: [], visible: [], labels: new Set(), checked: new Set(), selected: null,
        onlyChecked: false, completed: false, elapsed: 0, refinement: null,
        supported: new Set(['multi-class']), maxCrops: 32,
        controller: null, masks: new Map(),
    };
    let tags = ['person'], preTags = ['person'], toastTimer;
    const palette = ['#ef4444', '#22c55e', '#3b82f6', '#f59e0b', '#a855f7', '#06b6d4', '#f97316', '#ec4899', '#84cc16', '#6366f1'];
    function colorFor(label) {
        let hash = 0;
        for (const char of label) hash = ((hash << 5) - hash + char.charCodeAt(0)) | 0;
        return palette[Math.abs(hash) % palette.length];
    }
    function rgbFor(color) { return [1, 3, 5].map(start => parseInt(color.slice(start, start + 2), 16)); }
    function node(tag, className, text) {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (text !== undefined) element.textContent = text;
        return element;
    }
    function toast(message, error = false) {
        clearTimeout(toastTimer);
        $('toast').textContent = message;
        $('toast').className = `toast ${error ? 'error' : 'success'}`;
        $('toast').style.display = 'block';
        toastTimer = setTimeout(() => { $('toast').style.display = 'none'; }, 5000);
    }

    function renderTags(predefined = false) {
        const container = $(predefined ? 'predefinedTagList' : 'tagList');
        const values = predefined ? preTags : tags;
        container.replaceChildren();
        values.forEach((label, index) => {
            const chip = node('div', 'tag');
            chip.appendChild(node('span', '', label));
            const remove = node('button', '', '×');
            remove.type = 'button';
            remove.setAttribute('aria-label', `移除标签 ${label}`);
            remove.addEventListener('click', () => { values.splice(index, 1); renderTags(predefined); });
            chip.appendChild(remove); container.appendChild(chip);
        });
    }
    function addTags(predefined = false) {
        const input = $(predefined ? 'predefinedTagAdd' : 'tagAdd');
        const values = predefined ? preTags : tags;
        if (!input.value.trim()) return;
        const combined = normalizeLabels([...values, input.value]);
        if (combined.length > 32) { toast('最多支持 32 个标签', true); return; }
        if (predefined) preTags = combined; else tags = combined;
        input.value = ''; renderTags(predefined);
    }
    ['tagAdd', 'predefinedTagAdd'].forEach((id, index) => {
        $(id).maxLength = 256;
        $(id).addEventListener('keydown', event => {
            if (event.key === 'Enter' && !event.isComposing) { event.preventDefault(); addTags(Boolean(index)); }
        });
    });

    function updateMode() {
        const refine = $('modeSelect').value === 'obj-refine';
        $('predefinedLabelsGroup').hidden = !refine;
        $('cropConfigGroup').hidden = !refine;
        $('mergeResults').disabled = !refine;
        $('tagInputLabel').textContent = refine ? '精细检测标签（按 Enter 添加）' : '识别标签（按 Enter 添加）';
        $('tagAdd').placeholder = refine ? '输入精细检测标签后回车' : '输入标签后回车';
        $('modeHint').textContent = refine
            ? '先通过预检测标签定位主体并裁剪，再用精细检测标签识别局部小目标；更多裁剪通常意味着更长耗时。'
            : '输入一个或多个类别名称，后端会对每个类别单独推理。';
        $('mergeResultsHint').textContent = refine
            ? '关闭后仍返回预检测和裁剪结果，但不再检测原图上的精细类别。'
            : '仅在 obj-refine 模式下生效';
    }
    $('modeSelect').addEventListener('change', updateMode);
    function toggleCropPanel() {
        $('cropConfigPanel').hidden = !$('cropConfigPanel').hidden;
        $('cropConfigToggle').setAttribute('aria-expanded', String(!$('cropConfigPanel').hidden));
        $('cropConfigToggle').lastElementChild.textContent = $('cropConfigPanel').hidden ? '▼' : '▲';
    }
    $('cropConfigToggle').addEventListener('click', toggleCropPanel);
    $('cropConfigToggle').addEventListener('keydown', event => {
        if (['Enter', ' '].includes(event.key)) { event.preventDefault(); toggleCropPanel(); }
    });

    function clearResults() {
        state.results = []; state.visible = []; state.labels.clear(); state.checked.clear();
        state.selected = null; state.onlyChecked = false; state.completed = false;
        state.refinement = null; state.masks.clear();
        $('jsonOutput').value = ''; $('stats').textContent = '';
        $('refinementStats').hidden = true;
        ['resultContainer', 'resultListPanel', 'filterPanel', 'scoreFilterPanel', 'areaFilterPanel'].forEach(id => { $(id).hidden = true; });
        $('placeholder').hidden = false; $('downloadBtn').disabled = true;
        $('toggleShowChecked').textContent = '只展示已勾选';
        $('toggleShowChecked').className = 'secondary';
    }

    function cancelPending() {
        ++state.requestVersion;
        if (state.controller) state.controller.abort();
        state.controller = null;
        $('submitBtn').disabled = false; $('submitBtn').textContent = '开始检测';
    }
    function releaseImage() {
        if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
        state.imageUrl = null; state.image = null; state.file = null;
    }
    async function handleFile(file) {
        if (!file) return;
        if (!file.type.startsWith('image/') && !/\.(jpe?g|png|bmp|webp|tiff?)$/i.test(file.name)) {
            toast('请选择图片文件', true); return;
        }
        if (file.size > 45 * 1024 * 1024) { toast('图片不能超过 45 MB', true); return; }
        cancelPending(); clearResults(); releaseImage();
        $('previewImage').removeAttribute('src');
        $('targetThumbBox').hidden = true; $('dropZone').hidden = false;
        const version = ++state.loadVersion;
        const image = new Image(), url = URL.createObjectURL(file);
        state.imageUrl = url;
        try {
            await new Promise((resolve, reject) => {
                image.onload = resolve; image.onerror = () => reject(new Error('无法读取图片，请选择其他图片'));
                image.src = url;
            });
            if (version !== state.loadVersion) return;
            state.file = file; state.image = image;
            $('previewImage').src = url;
            $('targetThumbBox').hidden = false; $('dropZone').hidden = true;
        } catch (error) {
            if (version === state.loadVersion) {
                releaseImage(); $('imageInput').value = ''; toast(error.message, true);
            }
        }
    }
    $('dropZone').addEventListener('click', event => { if (event.target !== $('imageInput')) $('imageInput').click(); });
    $('dropZone').addEventListener('keydown', event => {
        if (['Enter', ' '].includes(event.key)) { event.preventDefault(); $('imageInput').click(); }
    });
    $('dropZone').addEventListener('dragover', event => { event.preventDefault(); $('dropZone').classList.add('dragover'); });
    $('dropZone').addEventListener('dragleave', () => $('dropZone').classList.remove('dragover'));
    $('dropZone').addEventListener('drop', event => {
        event.preventDefault(); $('dropZone').classList.remove('dragover'); handleFile(event.dataTransfer.files[0]);
    });
    $('imageInput').addEventListener('change', event => handleFile(event.target.files[0]));
    $('reuploadTargetBtn').addEventListener('click', () => {
        ++state.loadVersion; cancelPending(); clearResults(); releaseImage();
        $('imageInput').value = ''; $('previewImage').removeAttribute('src');
        $('targetThumbBox').hidden = true; $('dropZone').hidden = false;
    });

    function renderFilters() {
        state.labels = new Set(state.results.map(item => item.label));
        $('displayConfidence').value = '0'; $('displayConfidenceValue').textContent = '0.00';
        $('minArea').value = '0'; $('filterList').replaceChildren();
        state.labels.forEach(label => {
            const chip = node('button', 'filter-chip'); chip.type = 'button'; chip.dataset.label = label;
            chip.setAttribute('aria-pressed', 'true');
            const dot = node('span', 'chip-dot'); dot.style.background = colorFor(label);
            chip.append(dot, node('span', '', label), node('span', 'chip-check', '✓'));
            chip.addEventListener('click', () => {
                if (state.labels.has(label)) state.labels.delete(label); else state.labels.add(label);
                refreshChips(); renderResult();
            });
            $('filterList').appendChild(chip);
        });
        ['filterPanel', 'scoreFilterPanel', 'areaFilterPanel'].forEach(id => { $(id).hidden = state.results.length === 0; });
        refreshChips();
    }
    function refreshChips() {
        $('filterList').querySelectorAll('.filter-chip').forEach(chip => {
            const active = state.labels.has(chip.dataset.label), color = colorFor(chip.dataset.label);
            chip.classList.toggle('inactive', !active);
            chip.setAttribute('aria-pressed', String(active));
            chip.style.borderColor = color; chip.style.color = color;
            chip.style.background = `rgba(${rgbFor(color).join(',')},0.12)`;
            chip.querySelector('.chip-check').style.background = color;
        });
    }
    $('selectAllBtn').addEventListener('click', () => { state.labels = new Set(state.results.map(item => item.label)); refreshChips(); renderResult(); });
    $('selectNoneBtn').addEventListener('click', () => { state.labels.clear(); refreshChips(); renderResult(); });

    function updateStats() {
        $('stats').textContent = `检测到 ${state.results.length} 个目标，当前展示 ${state.visible.length} 个，已勾选 ${state.checked.size} 个 · ${(state.elapsed / 1000).toFixed(2)} s`;
        $('clearCheckedBtn').hidden = state.checked.size === 0;
        const metadata = state.refinement;
        $('refinementStats').hidden = !metadata;
        if (metadata) {
            $('refinementStats').textContent = `裁剪 ${metadata.crops_processed}/${metadata.candidate_crops} 个区域` +
                (metadata.limited ? ' · 已触及预算，部分区域未执行局部检测' : '') +
                ` · 后端 ${Number(metadata.timings_ms?.total || 0).toFixed(0)} ms`;
            $('refinementStats').style.color = metadata.limited ? '#fcd34d' : '#9ca3af';
        }
    }

    function maskCanvas(item, index) {
        if (state.masks.has(index)) return state.masks.get(index);
        const width = item.mask_width, height = item.mask_height;
        const canvas = document.createElement('canvas');
        const pixels = decodeRle(item.mask, width, height, rgbFor(colorFor(item.label)));
        canvas.width = width; canvas.height = height;
        const context = canvas.getContext('2d'), image = context.createImageData(width, height);
        image.data.set(pixels); context.putImageData(image, 0, 0);
        state.masks.set(index, canvas);
        return canvas;
    }
    function renderResult() {
        if (!state.completed || !state.image) return;
        const width = state.image.naturalWidth, height = state.image.naturalHeight;
        $('sourceImage').src = state.imageUrl;
        const canvas = $('overlayCanvas');
        if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
        const context = canvas.getContext('2d'); context.clearRect(0, 0, width, height);
        state.visible = visibleResults(state.results, {
            labels: state.labels, confidence: Number($('displayConfidence').value),
            area: Math.max(0, Number($('minArea').value) || 0), onlyChecked: state.onlyChecked,
            checked: state.checked, sort: $('sortBy').value,
        });
        // Selected item draws last, without drawing the same mask twice.
        const paint = [...state.visible].sort((a, b) => Number(a.index === state.selected) - Number(b.index === state.selected));
        for (const { item, index } of paint) {
            const [x1, y1, x2, y2] = item.box, color = colorFor(item.label), selected = index === state.selected;
            if ($('showMasks').checked && Array.isArray(item.mask) && item.mask.length) {
                try {
                    const mask = maskCanvas(item, index);
                    context.save(); context.globalAlpha = selected ? 0.65 : 0.45;
                    // Native masks are aligned to the integer bbox origin, not stretched to float bounds.
                    context.drawImage(mask, Math.floor(x1), Math.floor(y1)); context.restore();
                } catch (error) { console.warn('Mask rendering skipped:', error.message); }
            }
            if ($('showBoxes').checked) {
                context.strokeStyle = color; context.lineWidth = Math.max(selected ? 4 : 2, width / 400);
                context.strokeRect(x1, y1, x2 - x1, y2 - y1);
                if (state.checked.has(index)) {
                    context.strokeStyle = '#facc15'; context.lineWidth = Math.max(3, width / 300);
                    context.strokeRect(x1 - 2, y1 - 2, x2 - x1 + 4, y2 - y1 + 4);
                }
                if (selected) {
                    context.strokeStyle = '#fff'; context.lineWidth = Math.max(2, width / 500);
                    context.strokeRect(x1 - 3, y1 - 3, x2 - x1 + 6, y2 - y1 + 6);
                }
            }
            if ($('showLabels').checked && $('showBoxes').checked) {
                const text = `${item.label} ${(item.score * 100).toFixed(1)}%`, fontSize = Math.max(12, width / 60);
                context.font = `bold ${fontSize}px sans-serif`;
                const textHeight = Math.max(16, width / 50), textWidth = context.measureText(text).width + 10;
                const y = Math.max(0, y1 - textHeight), x = Math.max(0, Math.min(x1, width - textWidth));
                context.fillStyle = color; context.fillRect(x, y, textWidth, textHeight);
                context.fillStyle = '#fff'; context.fillText(text, x + 5, y + textHeight - 4);
            }
        }
        $('legend').replaceChildren();
        new Set(state.visible.map(({ item }) => item.label)).forEach(label => {
            const legend = node('div', 'legend-item'), swatch = node('span', 'legend-color');
            swatch.style.background = colorFor(label); legend.append(swatch, node('span', '', label)); $('legend').appendChild(legend);
        });
        renderList(); updateStats();
        $('resultContainer').hidden = false; $('placeholder').hidden = true; $('downloadBtn').disabled = false;
    }
    function renderList() {
        $('resultListPanel').hidden = false; $('detectionList').replaceChildren();
        if (!state.visible.length) {
            $('detectionList').appendChild(node('div', 'empty-state', state.onlyChecked
                ? '当前没有符合筛选条件的已勾选结果，或尚未勾选任何目标。'
                : (state.results.length ? '当前筛选条件无匹配结果。' : '没有检测到目标。')));
            return;
        }
        const grid = node('div', 'detection-grid');
        state.visible.forEach(({ item, index }, number) => {
            const card = node('div', 'detection-card');
            card.classList.toggle('selected', index === state.selected); card.classList.toggle('checked', state.checked.has(index));
            card.tabIndex = 0;
            const heading = node('div', 'card-heading'), check = node('input', 'result-check');
            check.type = 'checkbox'; check.checked = state.checked.has(index);
            check.setAttribute('aria-label', `勾选目标 ${number + 1} ${item.label}`);
            const title = node('span', 'card-label'), dot = node('span', 'card-dot'); dot.style.background = colorFor(item.label);
            title.append(dot, document.createTextNode(`#${number + 1} ${item.label}`)); heading.append(check, title);
            card.append(heading, node('div', 'card-detail', `${(item.score * 100).toFixed(0)}% · ${Math.round(resultArea(item)).toLocaleString()}px²`),
                node('div', 'card-coords', `[${item.box.map(value => Math.round(value)).join(', ')}]`));
            check.addEventListener('click', event => { event.stopPropagation();
                if (check.checked) state.checked.add(index); else state.checked.delete(index); renderResult(); });
            const select = () => { state.selected = index; renderResult(); };
            card.addEventListener('click', select);
            card.addEventListener('keydown', event => {
                if (event.target === card && ['Enter', ' '].includes(event.key)) { event.preventDefault(); select(); }
            });
            grid.appendChild(card);
        });
        $('detectionList').appendChild(grid);
    }

    $('confidence').addEventListener('input', () => { $('confidenceValue').textContent = Number($('confidence').value).toFixed(2); });
    $('displayConfidence').addEventListener('input', () => {
        $('displayConfidenceValue').textContent = Number($('displayConfidence').value).toFixed(2); renderResult();
    });
    ['showBoxes', 'showMasks', 'showLabels', 'sortBy'].forEach(id => $(id).addEventListener('change', renderResult));
    $('minArea').addEventListener('input', renderResult);
    $('toggleShowChecked').addEventListener('click', () => {
        state.onlyChecked = !state.onlyChecked;
        $('toggleShowChecked').textContent = state.onlyChecked ? '展示全部结果' : '只展示已勾选';
        $('toggleShowChecked').className = state.onlyChecked ? 'active' : 'secondary'; renderResult();
    });
    $('clearCheckedBtn').addEventListener('click', () => {
        state.checked.clear(); state.onlyChecked = false;
        $('toggleShowChecked').textContent = '只展示已勾选'; $('toggleShowChecked').className = 'secondary'; renderResult();
    });
    $('overlayCanvas').addEventListener('click', event => {
        const rect = $('overlayCanvas').getBoundingClientRect();
        const x = (event.clientX - rect.left) * $('overlayCanvas').width / rect.width;
        const y = (event.clientY - rect.top) * $('overlayCanvas').height / rect.height;
        const hit = [...state.visible].reverse().find(({ item }) => x >= item.box[0] && x <= item.box[2] && y >= item.box[1] && y <= item.box[3]);
        if (hit) { state.selected = hit.index; renderResult(); $('detectionList').querySelector('.selected')?.scrollIntoView({ block: 'nearest' }); }
    });
    $('downloadBtn').addEventListener('click', () => {
        if (!state.completed || !state.image) return;
        const canvas = document.createElement('canvas'); canvas.width = state.image.naturalWidth; canvas.height = state.image.naturalHeight;
        const context = canvas.getContext('2d'); context.drawImage(state.image, 0, 0); context.drawImage($('overlayCanvas'), 0, 0);
        canvas.toBlob(blob => {
            if (!blob) { toast('结果图导出失败', true); return; }
            const url = URL.createObjectURL(blob), link = node('a'); link.href = url; link.download = `sam3-result-${Date.now()}.png`;
            link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
        }, 'image/png');
    });

    function collectOptions() {
        const fieldIds = { max_size: 'cropMaxSize', padding: 'cropPadding', w_diou: 'cropWDiou',
            w_expansion: 'cropWExpansion', count_penalty: 'cropCountPenalty', nms_threshold: 'cropNmsThreshold',
            target_ar: 'cropTargetAr', max_crops: 'cropMaxCrops' };
        const crop = Object.fromEntries(Object.entries(fieldIds).map(([key, id]) => [key, $(id).value]));
        crop.enable_ar_fix = $('cropEnableAr').checked;
        return { mode: $('modeSelect').value, labels: tags, preLabels: preTags, confidence: $('confidence').value,
            returnMask: $('returnMask').checked, mergeResults: $('mergeResults').checked,
            preConfidence: $('preDetectConfidence').value, crop, maxCrops: state.maxCrops };
    }
    $('submitBtn').addEventListener('click', async () => {
        if (!state.file || !state.image) { toast('请先选择目标图片', true); return; }
        addTags(); addTags(true); // commit pending text even without Enter
        let request;
        try {
            if (!state.supported.has($('modeSelect').value)) throw new Error('后端尚未支持或未确认这个模式');
            request = buildInferenceForm(state.file, collectOptions());
        } catch (error) { toast(error.message, true); return; }
        const version = ++state.requestVersion, file = state.file;
        state.controller = new AbortController();
        $('submitBtn').disabled = true; $('submitBtn').textContent = '检测中...';
        const started = performance.now();
        try {
            const response = await fetch(request.endpoint, { method: 'POST', body: request.formData, signal: state.controller.signal });
            const type = response.headers.get('content-type') || '';
            if (!type.includes('application/json')) throw new Error(`服务返回非 JSON 响应（HTTP ${response.status}），请检查网关和后端日志`);
            const data = await response.json();
            if (!response.ok || data.error) throw new Error(formatError(data, response.status));
            if (!Array.isArray(data.results)) throw new Error('响应缺少 results 数组');
            if (version !== state.requestVersion || file !== state.file) return;
            clearResults(); state.elapsed = performance.now() - started;
            state.results = data.results.filter(item => typeof item.label === 'string' && Number.isFinite(item.score) &&
                Array.isArray(item.box) && item.box.length === 4 && item.box.every(Number.isFinite) && resultArea(item) > 0);
            state.refinement = data.refinement || null; state.completed = true;
            $('jsonOutput').value = JSON.stringify(data, null, 2);
            renderFilters(); renderResult(); toast(data.refinement?.limited ? '检测完成，部分裁剪区域因预算限制未执行' : '检测完成');
        } catch (error) {
            if (version === state.requestVersion && error.name !== 'AbortError') toast(`检测失败：${error.message}`, true);
        } finally {
            if (version === state.requestVersion) {
                state.controller = null; $('submitBtn').disabled = false; $('submitBtn').textContent = '开始检测';
            }
        }
    });

    async function loadCapabilities() {
        try {
            const response = await fetch('/ui-config', { cache: 'no-store' });
            if (!response.ok) throw new Error('后端未提供界面能力配置');
            const data = await response.json();
            state.supported = new Set(data.supported_modes.filter(mode => ['multi-class', 'obj-refine'].includes(mode)));
            if (state.supported.has('obj-refine')) {
                $('modeSelect').querySelector('[value="obj-refine"]').disabled = false;
                state.maxCrops = data.refinement_limits.max_crops;
                $('cropMaxCrops').max = String(state.maxCrops);
                $('cropMaxCrops').placeholder = `服务器默认/上限 ${state.maxCrops}`;
            }
            $('capabilityHint').textContent = 'Ascend 后端支持文本提示和目标细化；几何框、混合框和跨图参考提示尚未实现。';
        } catch (error) {
            $('capabilityHint').textContent = '暂时无法确认目标细化能力，仅启用文本检测；请确认后端镜像已更新。';
        }
    }
    window.addEventListener('beforeunload', () => { if (state.imageUrl) URL.revokeObjectURL(state.imageUrl); });
    renderTags(); renderTags(true); updateMode(); loadCapabilities();
})();
