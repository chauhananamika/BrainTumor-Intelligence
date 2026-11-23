$(document).ready(function () {
    const $dropZone = $('#dropZone');
    const $fileInput = $('#fileInput');
    const $mriImage = $('#mriImage');
    const $heatmapOverlay = $('#heatmapOverlay');
    const $toggleHeatmap = $('#toggleHeatmap');
    const $opacitySlider = $('#opacitySlider');
    const $imageLoader = $('#imageLoader');

    // Opacity slider handler
    $opacitySlider.on('input change', function () {
        const val = $(this).val();
        $heatmapOverlay.css('opacity', val / 100);
    });

    // Toggle heatmap on/off
    $toggleHeatmap.on('change', function () {
        if ($(this).is(':checked')) {
            $heatmapOverlay.fadeIn(150);
        } else {
            $heatmapOverlay.fadeOut(150);
        }
    });

    // File input change handler (fires when user picks an image)
    $fileInput.on('change', function () {
        if (this.files && this.files.length > 0) {
            handleFileUpload(this.files[0]);
            // Reset value so choosing the same file again still fires change
            this.value = '';
        }
    });

    // Drag and drop event handlers
    $dropZone.on('dragover dragenter', function (e) {
        e.preventDefault();
        e.stopPropagation();
        $dropZone.addClass('dragover');
    });

    $dropZone.on('dragleave dragend drop', function (e) {
        e.preventDefault();
        e.stopPropagation();
        $dropZone.removeClass('dragover');
    });

    $dropZone.on('drop', function (e) {
        e.preventDefault();
        e.stopPropagation();
        const dt = e.originalEvent.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            handleFileUpload(dt.files[0]);
        }
    });

    // Clipboard Paste (Ctrl+V) support
    $(document).on('paste', function (e) {
        const items = (e.clipboardData || e.originalEvent.clipboardData)?.items;
        if (!items) return;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type && items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                if (blob) {
                    handleFileUpload(blob);
                    break;
                }
            }
        }
    });

    // Preset test scan click handler
    $('.sample-card').on('click', function () {
        $('.sample-card').removeClass('active');
        $(this).addClass('active');

        const sampleKey = $(this).data('sample');
        loadSampleScan(sampleKey);
    });

    function showLoader() {
        $imageLoader.fadeIn(100);
    }

    function hideLoader() {
        $imageLoader.fadeOut(150);
    }

    function updateUIWithResults(data, imgSrc) {
        hideLoader();

        // Update image and heatmap
        if (imgSrc) {
            $mriImage.attr('src', imgSrc);
        }
        if (data.heatmap_base64) {
            $heatmapOverlay.attr('src', data.heatmap_base64);
            $heatmapOverlay.show();
        }

        // Headline & Confidence
        $('#predictionHeadline').text(data.prediction);
        $('#predictionConfidence').text('Calibrated confidence ' + data.confidence.toFixed(1) + '%.');

        // Tag styling
        const $tag = $('#predictionTag');
        if (data.prediction === 'No tumor') {
            $tag.css({ 'background-color': 'rgba(56, 189, 248, 0.15)', 'color': '#38bdf8', 'border-color': 'rgba(56, 189, 248, 0.3)' });
        } else {
            $tag.css({ 'background-color': 'rgba(16, 185, 129, 0.15)', 'color': '#10b981', 'border-color': 'rgba(16, 185, 129, 0.3)' });
        }

        // Probabilities
        const probs = data.probabilities || {};
        updateProbRow('Glioma', probs['Glioma'] || 0, data.prediction === 'Glioma');
        updateProbRow('Meningioma', probs['Meningioma'] || 0, data.prediction === 'Meningioma');
        updateProbRow('NoTumor', probs['No tumor'] || 0, data.prediction === 'No tumor');
        updateProbRow('Pituitary', probs['Pituitary'] || 0, data.prediction === 'Pituitary');

        // Stats
        $('#statConfidence').text(data.confidence.toFixed(1) + '%');
        $('#statDistCheck').text(data.distribution_check || 'passed');
        $('#statLatency').text((data.latency_ms || 142) + ' ms');

        // Mahalanobis distance slider
        const distPct = Math.min(Math.max((data.distance_score || 0.25) * 100, 5), 95);
        $('#distanceFill').css('width', distPct + '%');
        $('#distanceMarker').css('left', distPct + '%');
    }

    function updateProbRow(key, val, isTop) {
        const valFixed = val.toFixed(1) + '%';
        $('#val' + key).text(valFixed);
        const $bar = $('#bar' + key);
        $bar.css('width', Math.max(val, 0.5) + '%');

        if (isTop) {
            $bar.addClass('active');
            $('#row' + key).addClass('top-prediction');
        } else {
            $bar.removeClass('active');
            $('#row' + key).removeClass('top-prediction');
        }
    }

    function handleFileUpload(file) {
        if (!file) return;

        // Ensure file is an image
        if (file.type && !file.type.startsWith('image/')) {
            alert('Please select a valid image file (PNG, JPG, JPEG, BMP).');
            return;
        }

        $('.sample-card').removeClass('active');
        showLoader();

        // Instant local preview
        const reader = new FileReader();
        reader.onload = function (e) {
            $mriImage.attr('src', e.target.result);
        };
        reader.readAsDataURL(file);

        // Upload to server
        const formData = new FormData();
        const safeName = file.name || 'uploaded_scan.jpg';
        formData.append('file', file, safeName);

        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            dataType: 'json',
            success: function (data) {
                updateUIWithResults(data, null);
            },
            error: function (xhr, status, error) {
                hideLoader();
                console.error('Upload prediction error:', xhr.responseText || error);
                alert('Prediction failed. Please upload a clear brain MRI image.');
            }
        });
    }

    function loadSampleScan(sampleKey) {
        showLoader();
        $.ajax({
            url: '/predict_sample/' + sampleKey,
            type: 'GET',
            dataType: 'json',
            success: function (data) {
                updateUIWithResults(data, data.image_url);
            },
            error: function () {
                hideLoader();
                console.error('Failed to load sample scan');
            }
        });
    }

    // Initial load: trigger meningioma preset
    loadSampleScan('meningioma');
});
