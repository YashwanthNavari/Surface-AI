import React, { useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Maximize2, Minimize2, Sliders } from 'lucide-react';

export default function ImageViewer({
  originalSrc,
  heatmapSrc,
  overlaySrc,
  altText = 'Steel Surface Matrix',
}) {
  const [zoom, setZoom] = useState(1);
  const [viewMode, setViewMode] = useState('original'); // 'original' | 'heatmap' | 'overlay'
  const [opacity, setOpacity] = useState(0.65);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.5, 3.5));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.5, 1));
  const handleResetZoom = () => setZoom(1);

  // Active image based on view mode
  let displayImage = originalSrc;
  if (viewMode === 'heatmap' && heatmapSrc) {
    displayImage = heatmapSrc;
  } else if (viewMode === 'overlay' && overlaySrc) {
    displayImage = overlaySrc;
  }

  return (
    <div
      className="image-viewport-wrapper"
      style={
        isFullscreen
          ? {
              position: 'fixed',
              inset: 0,
              zIndex: 9999,
              borderRadius: 0,
              width: '100vw',
              height: '100vh',
            }
          : {}
      }
    >
      {/* Top Floating Viewport Actions */}
      <div className="viewport-overlay-controls">
        <button
          className="viewport-btn"
          onClick={handleZoomIn}
          title="Zoom In"
          disabled={zoom >= 3.5}
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          className="viewport-btn"
          onClick={handleZoomOut}
          title="Zoom Out"
          disabled={zoom <= 1}
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          className="viewport-btn"
          onClick={handleResetZoom}
          title="Reset Zoom"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
        <button
          className="viewport-btn"
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
            className="image-specimen-layer"
            style={{
              transform: `scale(${zoom})`,
            }}
          />

          {/* Optional Blended Overlay Layer with opacity slider */}
          {viewMode === 'overlay' && overlaySrc && (
            <img
              src={overlaySrc}
              alt="Grad-CAM Overlay Layer"
              className="image-specimen-layer"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                transform: `scale(${zoom})`,
                opacity: opacity,
                pointerEvents: 'none',
              }}
            />
          )}

          {/* Direct Heatmap Layer */}
          {viewMode === 'heatmap' && heatmapSrc && (
            <img
              src={heatmapSrc}
              alt="Grad-CAM Heatmap Layer"
              className="image-specimen-layer"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                transform: `scale(${zoom})`,
                pointerEvents: 'none',
              }}
            />
          )}
        </div>
      ) : (
        <div style={{ color: 'var(--text-dim)', fontSize: '13px' }}>No specimen acquired</div>
      )}

      {/* Bottom Floating Bar: View Modes & Opacity */}
      {originalSrc && (
        <div className="viewport-bottom-bar">
          <div className="segmented-control" style={{ background: 'rgba(255, 255, 255, 0.9)', backdropFilter: 'blur(8px)' }}>
            <button
              className={`segmented-btn ${viewMode === 'original' ? 'active' : ''}`}
              onClick={() => setViewMode('original')}
            >
              Original
            </button>
            {heatmapSrc && (
              <button
                className={`segmented-btn ${viewMode === 'heatmap' ? 'active' : ''}`}
                onClick={() => setViewMode('heatmap')}
              >
                Heatmap
              </button>
            )}
            {overlaySrc && (
              <button
                className={`segmented-btn ${viewMode === 'overlay' ? 'active' : ''}`}
                onClick={() => setViewMode('overlay')}
              >
                Overlay
              </button>
            )}
          </div>

          {viewMode === 'overlay' && overlaySrc && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                background: 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(8px)',
                border: '1px solid var(--border-default)',
                borderRadius: 'var(--radius-sm)',
                padding: '4px 10px',
                fontSize: '11px',
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-secondary)',
                boxShadow: 'var(--shadow-sm)',
              }}
            >
              <Sliders className="w-3.5 h-3.5 text-slate-500" />
              <span>Opacity</span>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                style={{ width: '60px', accentColor: 'var(--accent-blue)', cursor: 'pointer' }}
              />
              <span>{Math.round(opacity * 100)}%</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
