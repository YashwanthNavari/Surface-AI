import React, { useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Maximize2, Minimize2, Layers, Eye } from 'lucide-react';

export default function ImageInspectionViewer({
  originalSrc,
  heatmapSrc,
  overlaySrc,
  altText = 'Component Surface',
}) {
  const [zoom, setZoom] = useState(1);
  const [viewMode, setViewMode] = useState('original'); // 'original' | 'heatmap' | 'overlay'
  const [opacity, setOpacity] = useState(0.65);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.5, 3.5));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.5, 1));
  const handleResetZoom = () => setZoom(1);

  // Determine active display image
  let displayImage = originalSrc;
  if (viewMode === 'heatmap' && heatmapSrc) {
    displayImage = heatmapSrc;
  } else if (viewMode === 'overlay' && overlaySrc) {
    displayImage = overlaySrc;
  }

  return (
    <div
      className="inspection-viewport-container"
      style={
        isFullscreen
          ? {
              position: 'fixed',
              top: 0,
              left: 0,
              width: '100vw',
              height: '100vh',
              maxHeight: 'none',
              zIndex: 9999,
              borderRadius: 0,
            }
          : {}
      }
    >
      {/* Viewport Floating Toolbar */}
      <div className="viewport-toolbar">
        <button
          className="viewport-tool-btn"
          onClick={handleZoomIn}
          title="Zoom In"
          disabled={zoom >= 3.5}
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          className="viewport-tool-btn"
          onClick={handleZoomOut}
          title="Zoom Out"
          disabled={zoom <= 1}
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          className="viewport-tool-btn"
          onClick={handleResetZoom}
          title="Reset Zoom"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
        <button
          className="viewport-tool-btn"
          onClick={() => setIsFullscreen(!isFullscreen)}
          title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
        >
          {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
        </button>
      </div>

      {/* Surface Image Matrix Display */}
      {originalSrc ? (
        <div
          style={{
            width: '100%',
            height: '100%',
            overflow: 'hidden',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
          }}
        >
          {/* Base Layer: Original Image */}
          <img
            src={originalSrc}
            alt={altText}
            className="inspection-image-layer"
            style={{
              transform: `scale(${zoom})`,
              transition: 'transform 0.15s ease-out',
            }}
          />

          {/* Saliency Layer: Overlaid Heatmap with Dynamic Opacity */}
          {viewMode === 'overlay' && heatmapSrc && (
            <img
              src={heatmapSrc}
              alt="Heatmap overlay"
              className="inspection-image-layer"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                opacity: opacity,
                mixBlendMode: 'screen',
                transform: `scale(${zoom})`,
                transition: 'transform 0.15s ease-out',
              }}
            />
          )}

          {/* Dedicated Heatmap Mode */}
          {viewMode === 'heatmap' && heatmapSrc && (
            <img
              src={heatmapSrc}
              alt="Grad-CAM activation heatmap"
              className="inspection-image-layer"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                transform: `scale(${zoom})`,
                transition: 'transform 0.15s ease-out',
              }}
            />
          )}
        </div>
      ) : (
        <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem', fontFamily: 'var(--font-mono)' }}>
          NO COMPONENT MATRIX LOADED
        </div>
      )}

      {/* View Mode Switcher Pills */}
      {heatmapSrc && (
        <div className="viewport-mode-switcher">
          <button
            className={`mode-pill ${viewMode === 'original' ? 'active' : ''}`}
            onClick={() => setViewMode('original')}
          >
            Original
          </button>
          <button
            className={`mode-pill ${viewMode === 'heatmap' ? 'active' : ''}`}
            onClick={() => setViewMode('heatmap')}
          >
            Heatmap
          </button>
          <button
            className={`mode-pill ${viewMode === 'overlay' ? 'active' : ''}`}
            onClick={() => setViewMode('overlay')}
          >
            Overlay
          </button>
        </div>
      )}

      {/* Overlay Opacity Intensity Slider */}
      {viewMode === 'overlay' && heatmapSrc && (
        <div className="opacity-slider-box">
          <span>INTENSITY</span>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
          />
          <span>{Math.round(opacity * 100)}%</span>
        </div>
      )}
    </div>
  );
}
