// script.js (Full, Updated Version with New Features)
document.addEventListener('DOMContentLoaded', () => {
    let map;
    let vendorLayerGroup = L.featureGroup();
    let polygonLayerGroup = L.featureGroup();
    let coverageGridLayerGroup = L.featureGroup();
    let heatmapLayer;
    let tempLocationMarker = null; // Variable for the find-on-map marker
    let currentHeatmapType = 'none';
    let showVendorRadius = true;
    let vendorsAreVisible = true;
    let allVendorsData = [];
    let allPolygonsData = null;
    let allCoverageGridData = [];
    let initialFilterData = {};
    let lastHeatmapData = null;
    let currentRadiusModifier = 1.0;
    let currentRadiusMode = 'percentage';
    let currentRadiusFixed = 3.0;
    let marketingAreasOnTop = false;
    const API_BASE_URL = '/api';
    // --- DOM Elements ---
    const bodyEl = document.body;
    const daterangeStartEl = document.getElementById('daterange-start');
    const daterangeEndEl = document.getElementById('daterange-end');
    const cityEl = document.getElementById('city');
    const areaMainTypeEl = document.getElementById('area-main-type');
    const vendorCodesFilterEl = document.getElementById('vendor-codes-filter');
    const vendorVisibleEl = document.getElementById('vendor-visible');
    const vendorIsOpenEl = document.getElementById('vendor-is-open');
    const vendorRadiusToggleBtn = document.getElementById('vendor-radius-toggle');
    const radiusEdgeColorEl = document.getElementById('radius-edge-color');
    const radiusInnerColorEl = document.getElementById('radius-inner-color');
    const radiusInnerNoneEl = document.getElementById('radius-inner-none');
    const vendorMarkerSizeEl = document.getElementById('vendor-marker-size');
    const markerSizeValueEl = document.getElementById('marker-size-value');
    const areaFillColorEl = document.getElementById('area-fill-color');
    const areaFillNoneEl = document.getElementById('area-fill-none');
    const applyFiltersBtn = document.getElementById('apply-filters-btn');
    // New elements
    const vendorRadiusModifierEl = document.getElementById('vendor-radius-modifier');
    const radiusModifierValueEl = document.getElementById('radius-modifier-value');
    const btnResetRadius = document.getElementById('btn-reset-radius');
    
    // New radius mode elements
    const radiusModeSelector = document.getElementById('radius-mode-selector');
    const radiusPercentageControl = document.getElementById('radius-percentage-control');
    const radiusFixedControl = document.getElementById('radius-fixed-control');
    const vendorRadiusFixedEl = document.getElementById('vendor-radius-fixed');
    const radiusFixedValueEl = document.getElementById('radius-fixed-value');
    
    // Grid visualization elements
    const gridBlurEl = document.getElementById('grid-blur');
    const gridBlurValueEl = document.getElementById('grid-blur-value');
    const gridFadeEl = document.getElementById('grid-fade');
    const gridFadeValueEl = document.getElementById('grid-fade-value');
    const marketingAreasOnTopBtn = document.getElementById('marketing-areas-on-top');
    const gridVisualizationSection = document.getElementById('grid-visualization-section');
    // Lat/Lng finder elements
    const latFinderInputEl = document.getElementById('lat-finder-input');
    const lngFinderInputEl = document.getElementById('lng-finder-input');
    const btnFindLocation = document.getElementById('btn-find-location');
    
    const vendorAreaMainTypeEl = document.getElementById('vendor-area-main-type');
    let globalLoadingOverlayEl = document.getElementById('map-loading-overlay-wrapper');
    if (!globalLoadingOverlayEl) {
        globalLoadingOverlayEl = document.createElement('div');
        globalLoadingOverlayEl.id = 'map-loading-overlay-wrapper';
        document.body.appendChild(globalLoadingOverlayEl); 
    }
    const mapTypeButtons = {
        densityTotal: document.getElementById('btn-order-density-total'),
        densityOrganic: document.getElementById('btn-order-density-organic'),
        densityNonOrganic: document.getElementById('btn-order-density-non-organic'),
        userDensity: document.getElementById('btn-user-density-heatmap'),
        population: document.getElementById('btn-population-heatmap'),
        vendors: document.getElementById('btn-vendors-map'),
    };
    const btnToggleVendors = document.getElementById('btn-toggle-vendors');
    const btnClearHeatmap = document.getElementById('btn-clear-heatmap');
    const heatmapRadiusEl = document.getElementById('heatmap-radius');
    const heatmapRadiusValueEl = document.getElementById('heatmap-radius-value');
    const heatmapBlurEl = document.getElementById('heatmap-blur');
    const heatmapBlurValueEl = document.getElementById('heatmap-blur-value');
    const heatmapMaxValEl = document.getElementById('heatmap-max-val');
    const heatmapMaxValValueEl = document.getElementById('heatmap-max-val-value');
    const customFilterConfigs = {
        areaSubType: {
            button: document.getElementById('area-sub-type-filter-button'),
            panel: document.getElementById('area-sub-type-filter-panel'),
            paramName: 'area_sub_type_filter', defaultText: 'Select Sub Types', optionsData: []
        },
        businessLine: {
            button: document.getElementById('business-line-filter-button'),
            panel: document.getElementById('business-line-filter-panel'),
            paramName: 'business_lines', defaultText: 'Select Business Lines', optionsData: []
        },
        vendorStatus: { 
            button: document.getElementById('vendor-status-filter-button'),
            panel: document.getElementById('vendor-status-filter-panel'),
            paramName: 'vendor_status_ids', defaultText: 'Select Statuses', optionsData: []
        },
        vendorGrade: { 
            button: document.getElementById('vendor-grade-filter-button'),
            panel: document.getElementById('vendor-grade-filter-panel'),
            paramName: 'vendor_grades', defaultText: 'Select Grades', optionsData: []
        },
        vendorAreaSubType: {
            button: document.getElementById('vendor-area-sub-type-filter-button'),
            panel: document.getElementById('vendor-area-sub-type-filter-panel'),
            paramName: 'vendor_area_sub_type', defaultText: 'Select Sub Areas', optionsData: []
        }
    };
    let defaultVendorIcon = L.icon({
        iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
        iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
        iconSize: [12 * 2, 12 * (41/25) * 2],
        iconAnchor: [12, 12 * (41/25) * 2],
        popupAnchor: [0, -12 * (41/25) * 2],
        shadowSize: [12 * (41/25) * 2, 12 * (41/25) * 2],
    });
    function updateVendorIconSize(baseSize) { 
        const aspectRatio = 41/25;
        const iconWidth = parseInt(baseSize);
        const iconHeight = parseInt(iconWidth * aspectRatio);
        defaultVendorIcon = L.icon({
            iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
            iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
            iconSize: [iconWidth, iconHeight],
            iconAnchor: [iconWidth / 2, iconHeight],
            popupAnchor: [0, -iconHeight + 5],
            shadowSize: [iconWidth * 1.5, iconHeight * 1.5],
            shadowAnchor: [iconWidth / 3, iconHeight*1.4]
        });
        markerSizeValueEl.textContent = baseSize;
        if(vendorsAreVisible) redrawVendorMarkersAndRadii();
    }
    function init() {
        initMap();
        fetchInitialFilterData().then(() => {
            populateCitySelect();
            initializeCustomDropdowns();
            applyDefaultFilters(); 
            setupEventListeners();
            const today = new Date(); 
            const thirtyDaysAgo = new Date(new Date().setDate(today.getDate() - 30));
            daterangeStartEl.value = thirtyDaysAgo.toISOString().split('T')[0];
            daterangeEndEl.value = today.toISOString().split('T')[0];
            updateVendorIconSize(vendorMarkerSizeEl.value); 
            
            // Adjust default heatmap values for normalized data (0-100 range)
            heatmapRadiusEl.value = "25";
            heatmapRadiusValueEl.textContent = "25";
            heatmapBlurEl.value = "15";
            heatmapBlurValueEl.textContent = "15";
            heatmapMaxValEl.value = "1.0";
            heatmapMaxValValueEl.textContent = "1.0";
            
            // Add tooltips to help users understand the parameters
            heatmapRadiusEl.title = "Controls the spread of heat from each point (5-50 pixels)";
            heatmapBlurEl.title = "Controls smoothness of color transitions (1-40)";
            heatmapMaxValEl.title = "Sets when maximum color is reached (0.1-2.0)";
            
            // Initialize radius modifier
            vendorRadiusModifierEl.value = "100";
            radiusModifierValueEl.textContent = "100";
            currentRadiusModifier = 1.0;
            currentRadiusMode = 'percentage';
            currentRadiusFixed = 3.0;
            vendorRadiusFixedEl.value = "3";
            radiusFixedValueEl.textContent = "3";
            
            // Initialize grid visualization controls
            gridBlurEl.value = "0";
            gridBlurValueEl.textContent = "0";
            gridFadeEl.value = "100";
            gridFadeValueEl.textContent = "100";
            marketingAreasOnTop = false;
            marketingAreasOnTopBtn.classList.remove('active');
            gridVisualizationSection.style.display = 'none'; // Initially hide
            
            btnToggleVendors.textContent = vendorsAreVisible ? 'Vendors On' : 'Vendors Off';
            btnToggleVendors.classList.toggle('active', vendorsAreVisible);
            fetchAndDisplayMapData();
        }).catch(error => {
            console.error("Initialization failed:", error);
            showLoading(true, `Initialization failed: ${error.message}. Please refresh.`);
        });
    }
    function initMap() {
        const mapContainer = document.getElementById('map');
        if(mapContainer) mapContainer.innerHTML = ''; else { console.error("Map container ('map') not found!"); return; }
        map = L.map('map', { 
            preferCanvas: true,
            attributionControl: false 
        }).setView([35.7219, 51.3347], 11);
        // Create dedicated panes to control layer order and clickability
        map.createPane('polygonPane');
        map.getPane('polygonPane').style.zIndex = 450; 
        
        map.createPane('coverageGridPane');
        map.getPane('coverageGridPane').style.zIndex = 460;
        
        // Use the default shadow pane for radii to ensure they are behind polygons
        map.getPane('shadowPane').style.zIndex = 250;
        L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> © <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd', maxZoom: 20
        }).addTo(map);
        L.control.attribution({position: 'bottomleft'}).addTo(map);
        
        if (vendorsAreVisible) vendorLayerGroup.addTo(map);
        polygonLayerGroup.addTo(map);
        coverageGridLayerGroup.addTo(map);
    }
    async function fetchInitialFilterData() {
        showLoading(true, 'Fetching initial settings...');
        try {
            const response = await fetch(`${API_BASE_URL}/initial-data`);
            if (!response.ok) {
                const errorText = await response.text();
                showLoading(true, `Error fetching settings: ${response.status}. Please refresh.`);
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }
            initialFilterData = await response.json();
            if (initialFilterData.error) {
                showLoading(true, `Backend error (settings): ${initialFilterData.error}. Please refresh.`);
                 throw new Error(`Backend error (initial data): ${initialFilterData.error} - ${initialFilterData.details || ''}`);
            }
        } catch (error) {
            console.error("Failed to fetch initial filter data:", error);
            showLoading(true, `Failed to load settings: ${error.message}. Please refresh.`);
            throw error;
        }
    }
    
    function populateCitySelect() {
        if (!initialFilterData || !initialFilterData.cities) return;
        populateSelectWithOptions(cityEl, initialFilterData.cities.map(c => ({ value: c.name, text: c.name })), false);
        cityEl.value = "tehran";
    }
    function initializeCustomDropdowns() {
        if (!initialFilterData) return;
        customFilterConfigs.businessLine.optionsData = (initialFilterData.business_lines || [])
            .map(bl => ({ value: bl, text: bl, checked: false }));
        renderCustomDropdown(customFilterConfigs.businessLine);
        
        customFilterConfigs.vendorStatus.optionsData = (initialFilterData.vendor_statuses || [])
            .map(s => ({ value: String(s), text: `Status ${s}`, checked: false }));
        renderCustomDropdown(customFilterConfigs.vendorStatus);
        customFilterConfigs.vendorGrade.optionsData = (initialFilterData.vendor_grades || [])
            .map(g => ({ value: g, text: g, checked: false }));
        renderCustomDropdown(customFilterConfigs.vendorGrade);
        updateCityDependentCustomFilters(cityEl.value);
        updateVendorAreaSubTypeFilter();
    }
    function applyDefaultFilters() {
        const statusConfig = customFilterConfigs.vendorStatus;
        const status5Option = statusConfig.optionsData.find(opt => opt.value === "5");
        if (status5Option) status5Option.checked = true;
        updateCustomDropdownButtonText(statusConfig);
        const gradeConfig = customFilterConfigs.vendorGrade;
        ["A", "A+"].forEach(gradeValue => {
            const gradeOption = gradeConfig.optionsData.find(opt => opt.value === gradeValue);
            if (gradeOption) gradeOption.checked = true;
        });
        updateCustomDropdownButtonText(gradeConfig);
        
        const blConfig = customFilterConfigs.businessLine;
        const restaurantOption = blConfig.optionsData.find(opt => opt.value && opt.value.toLowerCase() === "restaurant");
        if (restaurantOption) restaurantOption.checked = true;
        updateCustomDropdownButtonText(blConfig);
        vendorVisibleEl.value = "1";
    }
    
    function updateCityDependentCustomFilters(selectedCity) {
        updateAreaSubTypeCustomFilter();
    }
    function updateAreaSubTypeCustomFilter() {
        if (!initialFilterData) return;
        const selectedAreaMainType = areaMainTypeEl.value;
        const selectedCity = cityEl.value;
        let subAreaOptionObjects = [];
        
        if (selectedAreaMainType === "coverage_grid") {
            // No sub-types for coverage grid
            customFilterConfigs.areaSubType.optionsData = [];
            renderCustomDropdown(customFilterConfigs.areaSubType);
            return;
        }
        
        if (selectedAreaMainType === "tapsifood_marketing_areas") {
            if (initialFilterData.marketing_areas_by_city && initialFilterData.marketing_areas_by_city[selectedCity]) {
                subAreaOptionObjects = initialFilterData.marketing_areas_by_city[selectedCity].map(name => ({ value: name, text: decodeURIComponentSafe(name) }));
            }
        } else if (selectedCity === "tehran") {
            if (selectedAreaMainType === "tehran_region_districts" && initialFilterData.tehran_region_districts) {
                subAreaOptionObjects = initialFilterData.tehran_region_districts.map(name => ({ value: name, text: decodeURIComponentSafe(name) }));
            } else if (selectedAreaMainType === "tehran_main_districts" && initialFilterData.tehran_main_districts) {
                subAreaOptionObjects = initialFilterData.tehran_main_districts.map(name => ({ value: name, text: decodeURIComponentSafe(name) }));
            } else if (selectedAreaMainType === "all_tehran_districts") {
                const regionD = (initialFilterData.tehran_region_districts || []).map(name => ({value: name, text: `Region: ${decodeURIComponentSafe(name)}`}));
                const mainD = (initialFilterData.tehran_main_districts || []).map(name => ({value: name, text: `Main: ${decodeURIComponentSafe(name)}`}));
                subAreaOptionObjects = [...regionD, ...mainD];
            }
        }
        customFilterConfigs.areaSubType.optionsData = subAreaOptionObjects
            .map(opt => ({ value: opt.value, text: opt.text, checked: false }));
        renderCustomDropdown(customFilterConfigs.areaSubType);
    }
    
    function updateVendorAreaSubTypeFilter() {
        if (!initialFilterData) return;
        const selectedVendorAreaType = vendorAreaMainTypeEl.value;
        const selectedCity = cityEl.value; 
        let subAreaOptionObjects = [];
        if (selectedVendorAreaType === "tapsifood_marketing_areas") {
             if (initialFilterData.marketing_areas_by_city && initialFilterData.marketing_areas_by_city[selectedCity]) {
                subAreaOptionObjects = initialFilterData.marketing_areas_by_city[selectedCity].map(name => ({ value: name, text: decodeURIComponentSafe(name) }));
            }
        } 
        else if (selectedVendorAreaType === "tehran_region_districts" && initialFilterData.tehran_region_districts) {
            subAreaOptionObjects = initialFilterData.tehran_region_districts.map(name => ({ value: name, text: decodeURIComponentSafe(name) }));
        } else if (selectedVendorAreaType === "tehran_main_districts" && initialFilterData.tehran_main_districts) {
            subAreaOptionObjects = initialFilterData.tehran_main_districts.map(name => ({ value: name, text: decodeURIComponentSafe(name) }));
        }
        
        customFilterConfigs.vendorAreaSubType.optionsData = subAreaOptionObjects
            .map(opt => ({ value: opt.value, text: opt.text, checked: false }));
        renderCustomDropdown(customFilterConfigs.vendorAreaSubType);
    }
    function decodeURIComponentSafe(str) {
        if (!str) return str;
        try {
            return decodeURIComponent(str.replace(/\+/g, ' ')); 
        } catch (e) {
            console.warn(`Failed to decode URI component: ${str}`, e);
            return str;
        }
    }
    
    function populateSelectWithOptions(selectElement, options, isMultiple = true) {
        selectElement.innerHTML = '';
        options.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.text;
            selectElement.add(option);
        });
    }
    function renderCustomDropdown(config) {
        config.panel.innerHTML = '';
        if (!config.optionsData || config.optionsData.length === 0) {
            config.panel.innerHTML = '<div class="dropdown-panel-item" style="color:var(--text-muted); cursor:default;">No options available</div>';
            updateCustomDropdownButtonText(config);
            return;
        }
        config.optionsData.forEach((option, index) => {
            const itemDiv = document.createElement('div');
            itemDiv.classList.add('dropdown-panel-item');
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `${config.paramName}-opt-${index}-${Math.random().toString(36).substr(2, 5)}`;
            checkbox.value = option.value;
            checkbox.checked = option.checked;
            const label = document.createElement('label');
            label.htmlFor = checkbox.id;
            label.textContent = option.text;
            checkbox.addEventListener('change', () => {
                option.checked = checkbox.checked;
                updateCustomDropdownButtonText(config);
            });
            itemDiv.addEventListener('click', (e) => {
                 if (e.target !== checkbox) checkbox.click();
            });
            itemDiv.appendChild(checkbox);
            itemDiv.appendChild(label);
            config.panel.appendChild(itemDiv);
        });
        updateCustomDropdownButtonText(config);
    }
    function updateCustomDropdownButtonText(config) {
        const selectedOptions = config.optionsData.filter(opt => opt.checked);
        const selectedCount = selectedOptions.length;
        if (selectedCount === 0) {
            config.button.textContent = config.defaultText;
        } else if (selectedCount === 1) {
            const selectedOptText = selectedOptions[0].text;
            config.button.textContent = selectedOptText.length > 20 ? selectedOptText.substring(0,18) + "..." : selectedOptText;
        } else {
            config.button.textContent = `${selectedCount} selected`;
        }
    }
    function getSelectedValuesFromCustomDropdown(config) {
        return config.optionsData.filter(opt => opt.checked).map(opt => opt.value);
    }
    function toggleDropdown(config, forceClose = false) {
        const isOpen = config.panel.classList.contains('open');
        if (forceClose || isOpen) {
            config.panel.classList.remove('open');
            config.button.classList.remove('open');
        } else {
            Object.values(customFilterConfigs).forEach(otherConfig => {
                if (otherConfig !== config) toggleDropdown(otherConfig, true);
            });
            config.panel.classList.add('open');
            config.button.classList.add('open');
        }
    }
    function setupEventListeners() {
        applyFiltersBtn.addEventListener('click', fetchAndDisplayMapData);
        cityEl.addEventListener('change', (e) => {
            updateCityDependentCustomFilters(e.target.value);
            updateVendorAreaSubTypeFilter();
        });
        areaMainTypeEl.addEventListener('change', () => {
            updateAreaSubTypeCustomFilter();
            // Show/hide grid visualization controls based on selection
            if (areaMainTypeEl.value === 'coverage_grid') {
                gridVisualizationSection.style.display = 'block';
            } else {
                gridVisualizationSection.style.display = 'none';
            }
        });
        vendorAreaMainTypeEl.addEventListener('change', updateVendorAreaSubTypeFilter);
        // Lat/Lng finder
        btnFindLocation.addEventListener('click', () => {
            const lat = parseFloat(latFinderInputEl.value);
            const lng = parseFloat(lngFinderInputEl.value);
            if (isNaN(lat) || isNaN(lng)) {
                alert('Please enter valid numbers for Latitude and Longitude.');
                return;
            }
            if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
                alert('Latitude must be between -90 and 90.\nLongitude must be between -180 and 180.');
                return;
            }
            // Remove previous temporary marker if it exists
            if (tempLocationMarker) {
                map.removeLayer(tempLocationMarker);
            }
            // Fly to the new location with a closer zoom
            map.flyTo([lat, lng], 16);
            // Add a distinct marker for the found location
            tempLocationMarker = L.circleMarker([lat, lng], {
                color: '#f44336', // Use a distinct red color
                radius: 10,
                weight: 3,
                opacity: 1,
                fillOpacity: 0.5,
                pane: 'markerPane' // Ensure it's on top
            }).bindPopup(`<b>Location</b><br>Lat: ${lat}<br>Lng: ${lng}`).addTo(map).openPopup();
        });
        
        // Updated map type buttons
        Object.keys(mapTypeButtons).forEach(type => {
            const button = mapTypeButtons[type];
            button.addEventListener('click', () => {
                const typeMapping = {
                    densityTotal: 'order_density',
                    densityOrganic: 'order_density_organic',
                    densityNonOrganic: 'order_density_non_organic',
                    userDensity: 'user_density',
                    population: 'population',
                    vendors: 'none'
                };
                currentHeatmapType = typeMapping[type] || 'none';
                setActiveMapTypeButton(type);
                fetchAndDisplayMapData();
            });
        });
        
        btnToggleVendors.addEventListener('click', () => {
            vendorsAreVisible = !vendorsAreVisible;
            btnToggleVendors.textContent = vendorsAreVisible ? 'Vendors On' : 'Vendors Off';
            btnToggleVendors.classList.toggle('active', vendorsAreVisible);
            if (vendorsAreVisible) {
                if (!map.hasLayer(vendorLayerGroup)) map.addLayer(vendorLayerGroup);
            } else {
                if (map.hasLayer(vendorLayerGroup)) map.removeLayer(vendorLayerGroup);
            }
        });
        btnClearHeatmap.addEventListener('click', () => {
            currentHeatmapType = 'none';
            setActiveMapTypeButton(null);
            if (heatmapLayer) { map.removeLayer(heatmapLayer); heatmapLayer = null; }
        });
        Object.values(customFilterConfigs).forEach(config => {
            config.button.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleDropdown(config);
            });
        });
        document.addEventListener('click', (e) => {
            Object.values(customFilterConfigs).forEach(config => {
                if (config.button && !config.button.contains(e.target) && config.panel && !config.panel.contains(e.target)) {
                    toggleDropdown(config, true);
                }
            });
        });
        vendorRadiusToggleBtn.addEventListener('click', () => {
            showVendorRadius = !showVendorRadius;
            vendorRadiusToggleBtn.textContent = showVendorRadius ? 'Hide Radius' : 'Show Radius';
            vendorRadiusToggleBtn.classList.toggle('active', showVendorRadius);
            redrawVendorRadii();
        });
        
        // Radius modifier controls
        radiusModeSelector.addEventListener('change', (e) => {
            currentRadiusMode = e.target.value;
            if (currentRadiusMode === 'fixed') {
                radiusPercentageControl.style.display = 'none';
                radiusFixedControl.style.display = 'block';
            } else {
                radiusPercentageControl.style.display = 'block';
                radiusFixedControl.style.display = 'none';
            }
        });
        
        vendorRadiusModifierEl.addEventListener('input', (e) => {
            const value = e.target.value;
            radiusModifierValueEl.textContent = value;
            currentRadiusModifier = parseInt(value) / 100;
        });
        
        vendorRadiusFixedEl.addEventListener('input', (e) => {
            const value = e.target.value;
            radiusFixedValueEl.textContent = value;
            currentRadiusFixed = parseFloat(value);
        });
        
        btnResetRadius.addEventListener('click', () => {
            vendorRadiusModifierEl.value = "100";
            radiusModifierValueEl.textContent = "100";
            currentRadiusModifier = 1.0;
            vendorRadiusFixedEl.value = "3";
            radiusFixedValueEl.textContent = "3";
            currentRadiusFixed = 3.0;
            fetchAndDisplayMapData(); // Re-fetch to apply original radius
        });
        
        // Grid visualization controls
        gridBlurEl.addEventListener('input', (e) => {
            gridBlurValueEl.textContent = e.target.value;
            applyGridVisualizationEffects();
        });
        
        gridFadeEl.addEventListener('input', (e) => {
            gridFadeValueEl.textContent = e.target.value;
            applyGridVisualizationEffects();
        });
        
        marketingAreasOnTopBtn.addEventListener('click', () => {
            marketingAreasOnTop = !marketingAreasOnTop;
            marketingAreasOnTopBtn.textContent = marketingAreasOnTop ? 'On' : 'Off';
            marketingAreasOnTopBtn.classList.toggle('active', marketingAreasOnTop);
            updateLayerOrder();
        });
        
        radiusEdgeColorEl.addEventListener('input', redrawVendorRadii);
        radiusInnerColorEl.addEventListener('input', redrawVendorRadii);
        radiusInnerNoneEl.addEventListener('change', redrawVendorRadii);
        areaFillColorEl.addEventListener('input', restylePolygons);
        areaFillNoneEl.addEventListener('change', restylePolygons);
        vendorMarkerSizeEl.addEventListener('input', (e) => updateVendorIconSize(e.target.value));
        heatmapRadiusEl.addEventListener('input', (e) => { heatmapRadiusValueEl.textContent = e.target.value; renderCurrentHeatmap(); });
        heatmapBlurEl.addEventListener('input', (e) => { heatmapBlurValueEl.textContent = e.target.value; renderCurrentHeatmap(); });
        heatmapMaxValEl.addEventListener('input', (e) => { heatmapMaxValValueEl.textContent = e.target.value; renderCurrentHeatmap(); });
    }
    
    function setActiveMapTypeButton(activeTypeKey) {
        Object.values(mapTypeButtons).forEach(btn => btn.classList.remove('active-map-type'));
        if (activeTypeKey && mapTypeButtons[activeTypeKey]) {
            mapTypeButtons[activeTypeKey].classList.add('active-map-type');
        }
    }
    function showLoading(isLoading, message = 'LOADING ...') {
        if (isLoading) {
            bodyEl.classList.add('is-loading');
            globalLoadingOverlayEl.textContent = message;
            globalLoadingOverlayEl.classList.add('visible');
        } else {
            bodyEl.classList.remove('is-loading');
            globalLoadingOverlayEl.classList.remove('visible');
        }
    }
    
    async function fetchAndDisplayMapData() {
        // Clear the temporary marker on new search
        if (tempLocationMarker) {
            map.removeLayer(tempLocationMarker);
            tempLocationMarker = null;
        }
        const params = new URLSearchParams();
        params.append('city', cityEl.value);
        params.append('start_date', daterangeStartEl.value);
        params.append('end_date', daterangeEndEl.value);
        params.append('area_type_display', areaMainTypeEl.value);
        getSelectedValuesFromCustomDropdown(customFilterConfigs.areaSubType)
            .forEach(val => params.append(customFilterConfigs.areaSubType.paramName, val));
        getSelectedValuesFromCustomDropdown(customFilterConfigs.businessLine)
            .forEach(val => params.append(customFilterConfigs.businessLine.paramName, val));
        const vendorCodesText = vendorCodesFilterEl.value.trim();
        if (vendorCodesText) params.append('vendor_codes_filter', vendorCodesText);
        params.append('vendor_area_main_type', vendorAreaMainTypeEl.value);
        getSelectedValuesFromCustomDropdown(customFilterConfigs.vendorAreaSubType)
            .forEach(val => params.append(customFilterConfigs.vendorAreaSubType.paramName, val));
        getSelectedValuesFromCustomDropdown(customFilterConfigs.vendorStatus)
            .forEach(val => params.append(customFilterConfigs.vendorStatus.paramName, val));
        getSelectedValuesFromCustomDropdown(customFilterConfigs.vendorGrade)
            .forEach(val => params.append(customFilterConfigs.vendorGrade.paramName, val));
        params.append('vendor_visible', vendorVisibleEl.value);
        params.append('vendor_is_open', vendorIsOpenEl.value);
        params.append('heatmap_type_request', currentHeatmapType);
        params.append('radius_modifier', currentRadiusModifier);
        params.append('radius_mode', currentRadiusMode);
        params.append('radius_fixed', currentRadiusFixed);
        console.log("Fetching map data with params:", params.toString());
        showLoading(true); 
        try {
            const response = await fetch(`${API_BASE_URL}/map-data?${params.toString()}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({error: "Unknown server error", details: response.statusText}));
                showLoading(true, `Error: ${errorData.error || response.statusText}. Check console.`);
                throw new Error(`HTTP error ${response.status}: ${errorData.error || response.statusText}`);
            }
            const data = await response.json();
            if(data.error){
                showLoading(true, `Backend error: ${data.error}. Check console.`);
                throw new Error(`Backend error: ${data.error} - ${data.details || ''}`);
            }
            allVendorsData = data.vendors || [];
            allPolygonsData = data.polygons || null;
            lastHeatmapData = data.heatmap_data || null; 
            allCoverageGridData = data.coverage_grid || [];
            
            // Debug logging for heatmap data
            if (currentHeatmapType !== 'none') {
                console.log(`Heatmap type: ${currentHeatmapType}, Points received: ${lastHeatmapData ? lastHeatmapData.length : 0}`);
                if (lastHeatmapData && lastHeatmapData.length > 0) {
                    console.log('Sample heatmap points:', lastHeatmapData.slice(0, 5));
                }
            }
            
            updateMapLayers();
            showLoading(false);
        } catch (error) {
            console.error("Fetch/Display Error:", error);
            if (!bodyEl.classList.contains('is-loading') || globalLoadingOverlayEl.textContent === 'LOADING ...') {
                 showLoading(true, `Error: ${error.message}. Check console or try refreshing.`);
            }
        }
    }
    function updateMapLayers() { 
        redrawVendorMarkersAndRadii(); 
        
        // Handle special case when grid is shown with marketing areas on top
        if (areaMainTypeEl.value === 'coverage_grid' && marketingAreasOnTop) {
            // Draw both grid and marketing areas
            drawCoverageGrid();
            
            // Temporarily show marketing areas
            const cityName = cityEl.value;
            if (cityName in gdf_marketing_areas) {
                // Save current polygon data
                const savedPolygonData = allPolygonsData;
                
                // Request marketing area polygons
                const params = new URLSearchParams();
                params.append('city', cityName);
                params.append('area_type_display', 'tapsifood_marketing_areas');
                
                // Fetch marketing area data synchronously (simplified for this case)
                // In production, this should be handled asynchronously
                if (initialFilterData.marketing_areas_by_city && initialFilterData.marketing_areas_by_city[cityName]) {
                    restylePolygons(); // This will use the existing polygon data
                }
            }
        } else {
            // Normal flow
            restylePolygons();
            drawCoverageGrid();
        }
        
        renderCurrentHeatmap(); 
        adjustMapView();
        applyGridVisualizationEffects(); // Apply visual effects after drawing
        updateLayerOrder(); // Ensure correct layer order
    }
    
    function renderCurrentHeatmap() {
        if (heatmapLayer) { map.removeLayer(heatmapLayer); heatmapLayer = null; }
        if (currentHeatmapType === 'none' || !lastHeatmapData || lastHeatmapData.length === 0) {
            return;
        }
        const heatPoints = lastHeatmapData.map(p => [p.lat, p.lng, p.value]);
        
        // Adjusted options for normalized data
        let heatOptions = { 
            radius: parseInt(heatmapRadiusEl.value), 
            blur: parseInt(heatmapBlurEl.value), 
            maxZoom: 18, 
            max: parseFloat(heatmapMaxValEl.value), // Now works well with normalized 0-100 values
            minOpacity: 0.3  // Add minimum opacity for better visibility
        };
        
        // Different gradients for different heatmap types
        if (currentHeatmapType === 'order_density' || currentHeatmapType === 'order_density_organic' || currentHeatmapType === 'order_density_non_organic') {
            // Order density heatmaps - cool to warm
            heatOptions.gradient = {
                0.0: 'rgba(0, 0, 255, 0)',
                0.2: 'rgba(0, 150, 255, 0.6)',
                0.4: 'rgba(0, 255, 255, 0.8)',
                0.6: 'rgba(0, 255, 0, 0.9)',
                0.8: 'rgba(255, 255, 0, 1)',
                1.0: 'rgba(255, 0, 0, 1)'
            };
        } else if (currentHeatmapType === 'user_density') {
            // User density - purple to orange
            heatOptions.gradient = {
                0.0: 'rgba(128, 0, 128, 0)',
                0.2: 'rgba(147, 112, 219, 0.6)',
                0.4: 'rgba(138, 43, 226, 0.8)',
                0.6: 'rgba(255, 165, 0, 0.9)',
                0.8: 'rgba(255, 140, 0, 1)',
                1.0: 'rgba(255, 69, 0, 1)'
            };
        } else {
            // Default gradient (population)
            heatOptions.gradient = {
                0.1: 'rgba(0,0,255,0)', 
                0.3: 'rgba(0, 255, 255, 1)', 
                0.5: 'rgba(0, 255, 0, 1)', 
                0.7: 'rgba(255, 255, 0, 1)', 
                1.0: 'rgba(255, 0, 0, 1)'
            };
        }
        
        heatmapLayer = L.heatLayer(heatPoints, heatOptions).addTo(map);
        // Make the heatmap canvas non-interactive so clicks pass through
        if (heatmapLayer && heatmapLayer.getPane()) {
            heatmapLayer.getPane().style.pointerEvents = 'none';
        }
    }
    
    function applyGridVisualizationEffects() {
        const blur = gridBlurEl.value;
        const fade = gridFadeEl.value / 100;
        
        // Apply CSS filter to the coverage grid pane
        const coveragePane = map.getPane('coverageGridPane');
        if (coveragePane) {
            coveragePane.style.filter = `blur(${blur}px)`;
            coveragePane.style.opacity = fade;
        }
    }
    
    function updateLayerOrder() {
        if (marketingAreasOnTop) {
            // Move marketing areas (polygons) on top of grid
            map.getPane('polygonPane').style.zIndex = 470;
            map.getPane('coverageGridPane').style.zIndex = 460;
        } else {
            // Default order - grid on top of polygons
            map.getPane('polygonPane').style.zIndex = 450;
            map.getPane('coverageGridPane').style.zIndex = 460;
        }
        
        // Force re-render of layers to apply new z-index
        if (polygonLayerGroup.getLayers().length > 0) {
            polygonLayerGroup.eachLayer(layer => {
                if (layer.setStyle) {
                    layer.setStyle(layer.options.style);
                }
            });
        }
        
        // Also update marketing areas visibility if grid is active
        if (areaMainTypeEl.value === 'coverage_grid' && marketingAreasOnTop) {
            // Show marketing areas as overlay
            const cityName = cityEl.value;
            if (cityName in gdf_marketing_areas && gdf_marketing_areas[cityName]) {
                // Temporarily switch to show marketing areas
                const currentAreaType = areaMainTypeEl.value;
                areaMainTypeEl.value = 'tapsifood_marketing_areas';
                restylePolygons();
                areaMainTypeEl.value = currentAreaType; // Switch back
            }
        }
    }
    
    function drawCoverageGrid() {
        coverageGridLayerGroup.clearLayers();
        
        if (areaMainTypeEl.value !== 'coverage_grid' || !allCoverageGridData || allCoverageGridData.length === 0) {
            return;
        }
        
        allCoverageGridData.forEach(point => {
            const coverage = point.coverage;
            const totalVendors = coverage.total_vendors;
            
            // Updated color coding based on requirements
            let color = '#ffffff'; // White for no coverage (should not appear as we filter out 0)
            if (totalVendors === 0) {
                color = '#ffffff'; // White
            } else if (totalVendors <= 3) {
                color = '#ff0000'; // Red
            } else if (totalVendors <= 6) {
                color = '#ff9900'; // Orange
            } else if (totalVendors <= 9) {
                color = '#ffff00'; // Yellow
            } else if (totalVendors <= 15) {
                color = '#00ff00'; // Green
            } else if (totalVendors <= 20) {
                color = '#006400'; // Dark Green
            } else {
                color = '#000000'; // Black for more than 20
            }
            
            // Create circle marker
            const marker = L.circleMarker([point.lat, point.lng], {
                radius: 6,
                fillColor: color,
                color: '#333',
                weight: 1,
                opacity: 0.8,
                fillOpacity: 0.8,
                pane: 'coverageGridPane'
            });
            
            // Create popup content
            let popupContent = `<b>Coverage at (${point.lat.toFixed(4)}, ${point.lng.toFixed(4)})</b><br>`;
            
            // Add marketing area if available
            if (point.marketing_area) {
                popupContent += `<b>Marketing Area:</b> ${decodeURIComponentSafe(point.marketing_area)}<br>`;
            }
            
            popupContent += `<b>Total Vendors:</b> ${totalVendors}<br>`;
            
            if (coverage.by_business_line && Object.keys(coverage.by_business_line).length > 0) {
                popupContent += '<b>By Business Line:</b><br>';
                Object.entries(coverage.by_business_line)
                    .sort((a, b) => b[1] - a[1]) // Sort by count descending
                    .forEach(([bl, count]) => {
                        popupContent += `&nbsp;&nbsp;${bl}: ${count}<br>`;
                    });
            }
            
            if (coverage.by_grade && Object.keys(coverage.by_grade).length > 0) {
                popupContent += '<b>By Grade:</b><br>';
                Object.entries(coverage.by_grade)
                    .sort((a, b) => {
                        // Sort grades with A+ first, then A, then others
                        const gradeOrder = {'A+': 1, 'A': 2};
                        const aOrder = gradeOrder[a[0]] || 999;
                        const bOrder = gradeOrder[b[0]] || 999;
                        if (aOrder !== bOrder) return aOrder - bOrder;
                        return b[1] - a[1]; // Then by count
                    })
                    .forEach(([grade, count]) => {
                        popupContent += `&nbsp;&nbsp;${grade}: ${count}<br>`;
                    });
            }
            
            marker.bindPopup(popupContent);
            coverageGridLayerGroup.addLayer(marker);
        });
    }
    
    function adjustMapView() {
        let bounds;
        const hasVisibleVendors = vendorsAreVisible && vendorLayerGroup.getLayers().length > 0;
        const hasVisiblePolygons = polygonLayerGroup.getLayers().length > 0;
        const hasVisibleCoverage = coverageGridLayerGroup.getLayers().length > 0;
        
        if (hasVisibleVendors) {
            bounds = vendorLayerGroup.getBounds();
            if (hasVisiblePolygons) {
                bounds.extend(polygonLayerGroup.getBounds());
            }
            if (hasVisibleCoverage) {
                bounds.extend(coverageGridLayerGroup.getBounds());
            }
        } else if (hasVisiblePolygons) {
            bounds = polygonLayerGroup.getBounds();
            if (hasVisibleCoverage) {
                bounds.extend(coverageGridLayerGroup.getBounds());
            }
        } else if (hasVisibleCoverage) {
            bounds = coverageGridLayerGroup.getBounds();
        }
        
        if (bounds && bounds.isValid()) {
            map.fitBounds(bounds, {padding: [50, 50]});
        } else if (cityEl.value === "tehran") {
            map.setView([35.7219, 51.3347], 11);
        } else if (cityEl.value === "mashhad") {
            map.setView([36.297, 59.606], 12);
        } else if (cityEl.value === "shiraz") {
            map.setView([29.5918, 52.5837], 12);
        }
    }
    function redrawVendorMarkersAndRadii() {
        vendorLayerGroup.clearLayers(); 
        if (!allVendorsData || allVendorsData.length === 0) return;
        allVendorsData.forEach(vendor => {
            if (vendor.latitude == null || vendor.longitude == null) return; 
            const popupContent = `<b>${vendor.vendor_name || 'N/A'}</b><br>
                                Code: ${vendor.vendor_code || 'N/A'}<br>
                                Status: ${vendor.status_id !== null ? vendor.status_id : 'N/A'}<br>
                                Grade: ${vendor.grade || 'N/A'}<br>
                                Visible: ${vendor.visible == 1 ? 'Yes' : (vendor.visible == 0 ? 'No' : 'N/A')}<br>
                                Open: ${vendor.open == 1 ? 'Yes' : (vendor.open == 0 ? 'No' : 'N/A')}<br>
                                Radius: ${vendor.radius ? vendor.radius.toFixed(2) + ' km' : 'N/A'}`;
            const marker = L.marker([vendor.latitude, vendor.longitude], {icon: defaultVendorIcon})
                .bindPopup(popupContent);
            vendorLayerGroup.addLayer(marker);
        });
        redrawVendorRadii(); 
    }
    
    function redrawVendorRadii() {
        vendorLayerGroup.eachLayer(layer => {
            if (layer instanceof L.Circle) {
                vendorLayerGroup.removeLayer(layer);
            }
        });
        if (!showVendorRadius || !allVendorsData) return; 
        const rEdgeColor = radiusEdgeColorEl.value;
        const rInnerIsNone = radiusInnerNoneEl.checked;
        const rInnerColor = rInnerIsNone ? 'transparent' : radiusInnerColorEl.value;
        allVendorsData.forEach(vendor => {
            if (vendor.latitude != null && vendor.longitude != null && vendor.radius > 0) {
                 L.circle([vendor.latitude, vendor.longitude], {
                    radius: vendor.radius * 1000, // radius is already modified by backend
                    color: rEdgeColor, 
                    fillColor: rInnerColor, 
                    fillOpacity: rInnerIsNone ? 0 : 0.25, 
                    weight: 1.5,
                    pane: 'shadowPane' // Render circles behind markers AND polygons
                }).addTo(vendorLayerGroup); 
            }
        });
    }
    function restylePolygons() {
        polygonLayerGroup.clearLayers();
        if (!allPolygonsData || !allPolygonsData.features || allPolygonsData.features.length === 0) return;
        const polyFillIsNone = areaFillNoneEl.checked;
        const polyFillColor = polyFillIsNone ? 'transparent' : areaFillColorEl.value;
        
        L.geoJSON(allPolygonsData, {
            pane: 'polygonPane',
            style: () => ({
                color: "#1E88E5", weight: 1.5, opacity: 0.7,
                fillColor: polyFillColor, fillOpacity: polyFillIsNone ? 0 : 0.3
            }),
            onEachFeature: (feature, layer) => {
                let popupContent = '';
                if (feature.properties) {
                    const p = feature.properties;
                    const nameCand = p.name || p.NAME || p.Name || p.Region || p.REGION_N || p.NAME_1 || p.NAME_2 || p.district || p.NAME_MAHAL;
                    const name = decodeURIComponentSafe(nameCand) || "Area Detail";
                    popupContent += `<b>${name}</b>`;
                    // Population Data Section (for Tehran districts)
                    if (p.Pop != null || p.PopDensity != null) {
                        popupContent += `<br><hr style="margin: 5px 0; border-color: #eee;"><em>Population Stats:</em>`;
                        if (p.Pop != null) {
                            popupContent += `<br><b>Population:</b> ${Number(p.Pop).toLocaleString()}`;
                        }
                        if (p.PopDensity != null) {
                             popupContent += `<br><b>Population Density:</b> ${Number(p.PopDensity).toFixed(2)}/km²`;
                        }
                    }
                    
                    // Filter-Based Data Section (applies to all polygon types)
                    if (p.vendor_count != null || p.unique_user_count != null) {
                        popupContent += `<br><hr style="margin: 5px 0; border-color: #eee;"><em>Metrics (based on filters):</em>`;
                        
                        // Vendor Metrics
                        if (p.vendor_count != null) {
                             popupContent += `<br><b>Total Filtered Vendors:</b> ${p.vendor_count}`;
                        }
                        if (p.grade_counts && Object.keys(p.grade_counts).length > 0) {
                            popupContent += `<br><b>- By Grade:</b> `;
                            const gradeStrings = Object.entries(p.grade_counts)
                                                      .sort((a, b) => b[1] - a[1]) // Sort by count desc
                                                      .map(([grade, count]) => `${grade}: ${count}`);
                            popupContent += gradeStrings.join(', ');
                        }
                        if (p.vendor_per_10k_pop != null) {
                            popupContent += `<br><b>- Vendors per 10k Pop:</b> ${Number(p.vendor_per_10k_pop).toFixed(2)}`;
                        }
                        // Customer Metrics
                        if (p.unique_user_count != null) {
                             popupContent += `<br><b>Unique Customers (date range):</b> ${p.unique_user_count.toLocaleString()}`;
                        }
                        if (p.total_unique_user_count != null) {
                             popupContent += `<br><b>Unique Customers (all time):</b> ${p.total_unique_user_count.toLocaleString()}`;
                        }
                    }
                } else {
                    popupContent = '<b>Area</b>';
                }
                layer.bindPopup(popupContent);
            }
        }).addTo(polygonLayerGroup);
    }
    init();
});