"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";

interface UploadProps {
  onUpload: (file: File, treatmentParams?: Record<string, any>) => void;
}

type Mode = "choose" | "camera" | "preview";

export default function Upload({ onUpload }: UploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [mode, setMode] = useState<Mode>("choose");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [facingMode, setFacingMode] = useState<"environment" | "user">("environment");
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nativeCameraRef = useRef<HTMLInputElement>(null);

  // ---------- Detect mobile ----------
  useEffect(() => {
    const checkMobile = () => {
      const ua = navigator.userAgent || "";
      const mobile = /Android|iPhone|iPad|iPod|webOS|BlackBerry|IEMobile|Opera Mini/i.test(ua)
        || (navigator.maxTouchPoints > 0 && window.innerWidth < 1024);
      setIsMobile(mobile);
    };
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  // ---------- Detect multiple cameras ----------
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices?.().then((devices) => {
      const videoDevices = devices.filter((d) => d.kind === "videoinput");
      setHasMultipleCameras(videoDevices.length > 1);
    }).catch(() => {});
  }, []);

  // ---------- Cleanup camera on unmount ----------
  useEffect(() => {
    return () => {
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------- File drop ----------
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setMode("preview");
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/jpeg": [".jpg", ".jpeg"],
      "image/png": [".png"],
      "image/webp": [".webp"],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  });

  // ---------- Native camera (mobile fallback) ----------
  const handleNativeCapture = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setMode("preview");
    }
    // Reset so the same file can be re-selected
    e.target.value = "";
  };

  // ---------- Camera (getUserMedia) ----------
  const startCamera = async (facing?: "environment" | "user") => {
    setCameraError(null);
    setCameraReady(false);
    stopCamera();

    const targetFacing = facing ?? facingMode;

    // On mobile, try native camera first if getUserMedia may be unreliable
    if (isMobile && !navigator.mediaDevices?.getUserMedia) {
      nativeCameraRef.current?.click();
      return;
    }

    try {
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: { ideal: targetFacing },
          width: { ideal: 1920, min: 640 },
          height: { ideal: 1080, min: 480 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // Wait for video to actually start playing
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().then(() => {
            setCameraReady(true);
          }).catch(() => {
            setCameraReady(true); // still show even if autoplay policy blocks
          });
        };
      }

      setFacingMode(targetFacing);
      setMode("camera");
    } catch (err: unknown) {
      const msg = err instanceof DOMException ? err.message : String(err);
      if (msg.includes("NotAllowedError") || msg.includes("Permission")) {
        setCameraError("Camera permission denied. Please allow camera access in your browser settings and try again.");
      } else if (msg.includes("NotFoundError") || msg.includes("DevicesNotFound")) {
        setCameraError("No camera found on this device.");
      } else if (msg.includes("NotReadableError") || msg.includes("TrackStartError")) {
        setCameraError("Camera is in use by another app. Close other camera apps and try again.");
      } else {
        setCameraError("Could not access camera. Try using the native camera option below.");
      }
      // On mobile, fall back to native camera input on error
      if (isMobile) {
        setTimeout(() => nativeCameraRef.current?.click(), 500);
      }
    }
  };

  const switchCamera = () => {
    const newFacing = facingMode === "environment" ? "user" : "environment";
    startCamera(newFacing);
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    // Use actual video resolution for max quality
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Mirror preview if using front camera
    if (facingMode === "user") {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (blob) {
          const file = new File([blob], `crop-scan-${Date.now()}.jpg`, { type: "image/jpeg" });
          setSelectedFile(file);
          setPreview(URL.createObjectURL(file));
          stopCamera();
          setMode("preview");
        }
      },
      "image/jpeg",
      0.92
    );
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraReady(false);
  };

  // ---------- Actions ----------
  const handleAnalyze = () => {
    if (!selectedFile) {
      setErrorMessage("Please upload or capture an image before analyzing.");
      return;
    }

    // Require either inorganic or organic selection
    if (!inorganicOption && !organicOption) {
      setErrorMessage("Please select either an Inorganic (chemical) or Organic option before analysis.");
      return;
    }

    // Require water amount
    const waterNum = Number(waterAmount);
    if (!waterAmount || isNaN(waterNum) || waterNum <= 0) {
      setErrorMessage("Please enter a valid Water amount in liters (e.g. 10).");
      return;
    }

    // If inorganic selected, require chemical amount and 'other' name when applicable
    if (inorganicOption) {
      if (inorganicOption === "other" && !inorganicOther.trim()) {
        setErrorMessage("Please specify the chemical name for 'Other' or choose a known chemical.");
        return;
      }
      const chemNum = Number(chemicalAmount);
      if (!chemicalAmount || isNaN(chemNum) || chemNum <= 0) {
        setErrorMessage("Please enter a valid Chemical amount in grams (e.g. 50) when using an inorganic selection.");
        return;
      }
    }

    // If organic selected, require organic amount and 'other' name when applicable
    if (organicOption) {
      if (organicOption === "other" && !organicOther.trim()) {
        setErrorMessage("Please specify the organic name for 'Other' or choose a known organic option.");
        return;
      }
      const orgNum = Number(organicAmount);
      if (!organicAmount || isNaN(orgNum) || orgNum <= 0) {
        setErrorMessage("Please enter a valid Organic amount in grams when using an organic selection.");
        return;
      }
    }

    // Passed validation
    setErrorMessage(null);
    const params: Record<string, any> = {};
    if (chemicalAmount) params.chemical_amount_g = Number(chemicalAmount);
    if (waterAmount) params.water_amount_l = Number(waterAmount);
    if (organicAmount) params.organic_amount_g = Number(organicAmount);
    if (notes) params.notes = notes;
    if (inorganicOption) params.inorganic = inorganicOption === "other" ? inorganicOther || "other" : inorganicOption;
    if (organicOption) params.organic = organicOption === "other" ? organicOther || "other" : organicOption;
    onUpload(selectedFile, Object.keys(params).length ? params : undefined);
  };

  // --- Calculator state ---
  const [chemicalAmount, setChemicalAmount] = useState<number | string>("");
  const [waterAmount, setWaterAmount] = useState<number | string>("");
  const [organicAmount, setOrganicAmount] = useState<number | string>("");
  const [notes, setNotes] = useState<string>("");
  const [inorganicOption, setInorganicOption] = useState<string>("");
  const [organicOption, setOrganicOption] = useState<string>("");
  const [inorganicOther, setInorganicOther] = useState<string>("");
  const [organicOther, setOrganicOther] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleClear = () => {
    if (preview) URL.revokeObjectURL(preview);
    stopCamera();
    setPreview(null);
    setSelectedFile(null);
    setMode("choose");
    setCameraError(null);
  };

  const openNativeCamera = () => {
    nativeCameraRef.current?.click();
  };

  return (
    <div className="card">
      {/* Hidden native camera input — reliable on all mobile devices */}
      <input
        ref={nativeCameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleNativeCapture}
        className="hidden"
        aria-hidden="true"
      />

      <h2 className="section-title flex items-center gap-2">
        <svg className="w-6 h-6 text-leaf-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
        </svg>
        Scan or Upload Leaf
      </h2>
      <p className="section-subtitle mb-6">
        {isMobile
          ? "Take a photo of the affected leaf or upload from your gallery"
          : "Use your camera to scan a leaf or upload an image file"}
      </p>


      {/* ── Choose mode ─────────────────────────────── */}
      {mode === "choose" && (
        <div className="space-y-4">
          <div className={`grid gap-4 ${isMobile ? "grid-cols-1" : "grid-cols-1 sm:grid-cols-2"}`}>

            {/* Camera Card — on mobile shows TWO options */}
            {isMobile ? (
              <>
                {/* Native camera (most reliable on mobile) */}
                <button
                  onClick={openNativeCamera}
                  className="flex items-center gap-4 p-5 border-2 border-dashed border-leaf-300 bg-leaf-50/30
                    rounded-2xl active:scale-[0.98] transition-all duration-200 group"
                >
                  <div className="w-14 h-14 rounded-2xl bg-leaf-100 flex items-center justify-center flex-shrink-0">
                    <svg className="w-7 h-7 text-leaf-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                        d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                      <circle cx="12" cy="13" r="3" strokeWidth={1.5} />
                    </svg>
                  </div>
                  <div className="text-left">
                    <p className="font-semibold text-leaf-700 text-base">Take Photo</p>
                    <p className="text-xs text-gray-500 mt-0.5">Opens your camera to capture a leaf</p>
                  </div>
                  <svg className="w-5 h-5 text-gray-400 ml-auto flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>

                {/* Live camera (getUserMedia) */}
                <button
                  onClick={() => startCamera()}
                  className="flex items-center gap-4 p-5 border-2 border-dashed border-gray-300
                    rounded-2xl active:scale-[0.98] transition-all duration-200 group"
                >
                  <div className="w-14 h-14 rounded-2xl bg-blue-50 flex items-center justify-center flex-shrink-0">
                    <svg className="w-7 h-7 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                        d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div className="text-left">
                    <p className="font-semibold text-gray-700 text-base">Live Camera</p>
                    <p className="text-xs text-gray-500 mt-0.5">Frame the leaf and capture in-app</p>
                  </div>
                  <svg className="w-5 h-5 text-gray-400 ml-auto flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </>
            ) : (
              /* Desktop: single camera button */
              <button
                onClick={() => startCamera()}
                className="flex flex-col items-center gap-3 p-8 border-2 border-dashed border-gray-300
                  rounded-2xl hover:border-leaf-400 hover:bg-leaf-50/50 transition-all duration-300 group"
              >
                <div className="w-14 h-14 rounded-2xl bg-leaf-100 group-hover:bg-leaf-200 flex items-center justify-center transition-colors">
                  <svg className="w-7 h-7 text-leaf-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    <circle cx="12" cy="13" r="3" strokeWidth={1.5} />
                  </svg>
                </div>
                <div className="text-center">
                  <p className="font-semibold text-gray-700 group-hover:text-leaf-700">Use Camera</p>
                  <p className="text-xs text-gray-400 mt-1">Take a live photo</p>
                </div>
              </button>
            )}

            {/* Upload Card */}
            <div
              {...getRootProps()}
              className={`flex ${isMobile ? "items-center gap-4 p-5" : "flex-col items-center gap-3 p-8"} border-2 border-dashed rounded-2xl
                cursor-pointer transition-all duration-300 group
                ${isDragActive
                  ? "border-leaf-500 bg-leaf-50 scale-[1.02]"
                  : "border-gray-300 hover:border-leaf-400 hover:bg-leaf-50/50"
                }
                ${isMobile ? "active:scale-[0.98]" : ""}`}
            >
              <input {...getInputProps()} />
              <div className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-colors flex-shrink-0
                ${isDragActive ? "bg-leaf-200" : "bg-gray-100 group-hover:bg-leaf-100"}`}>
                <svg className={`w-7 h-7 transition-colors ${isDragActive ? "text-leaf-600" : "text-gray-400 group-hover:text-leaf-600"}`}
                  fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <div className={isMobile ? "text-left" : "text-center"}>
                {isDragActive ? (
                  <p className="font-semibold text-leaf-600">Drop it here</p>
                ) : (
                  <>
                    <p className="font-semibold text-gray-700 group-hover:text-leaf-700">
                      {isMobile ? "Choose from Gallery" : "Upload Image"}
                    </p>
                    <p className="text-xs text-gray-400 mt-0.5 sm:mt-1">JPEG, PNG, WebP &middot; Max 10 MB</p>
                  </>
                )}
              </div>
              {isMobile && !isDragActive && (
                <svg className="w-5 h-5 text-gray-400 ml-auto flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              )}
            </div>
          </div>

          {cameraError && (
            <div className="bg-red-50 border border-red-200 text-red-600 text-sm rounded-xl px-4 py-3">
              <p>{cameraError}</p>
              {isMobile && (
                <button onClick={openNativeCamera}
                  className="mt-2 text-leaf-600 font-medium underline underline-offset-2 text-sm">
                  Use native camera instead
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Camera mode ─────────────────────────────── */}
      {mode === "camera" && (
        <div className="space-y-4">
          <div className="relative rounded-2xl overflow-hidden bg-black border border-gray-200">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full object-cover ${isMobile ? "h-[60vh] max-h-[500px]" : "h-64 sm:h-80"} ${
                facingMode === "user" ? "scale-x-[-1]" : ""
              }`}
            />

            {/* Loading spinner while camera initializes */}
            {!cameraReady && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-10 h-10 border-3 border-white/30 border-t-white rounded-full animate-spin" />
                  <p className="text-white text-sm">Starting camera...</p>
                </div>
              </div>
            )}

            {/* Viewfinder overlay */}
            {cameraReady && (
              <div className="absolute inset-0 pointer-events-none">
                {/* Corner brackets */}
                <div className="absolute inset-8 sm:inset-12">
                  <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-white/60 rounded-tl-lg" />
                  <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-white/60 rounded-tr-lg" />
                  <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-white/60 rounded-bl-lg" />
                  <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-white/60 rounded-br-lg" />
                </div>
                <div className="absolute top-3 left-3 bg-black/50 text-white text-xs px-2 py-1 rounded-lg flex items-center gap-1.5">
                  <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                  Live
                </div>
                <p className="absolute bottom-3 left-0 right-0 text-center text-white/70 text-xs">
                  Position the leaf inside the frame
                </p>
              </div>
            )}

            {/* Camera switch button */}
            {hasMultipleCameras && cameraReady && (
              <button
                onClick={switchCamera}
                className="absolute top-3 right-3 w-10 h-10 bg-black/50 backdrop-blur-sm rounded-full
                  flex items-center justify-center text-white active:scale-90 transition-transform"
                aria-label="Switch camera"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            )}
          </div>
          <canvas ref={canvasRef} className="hidden" />

          {/* Mobile-optimized capture controls */}
          <div className={`flex items-center ${isMobile ? "justify-center gap-6 py-2" : "gap-3"}`}>
            {isMobile ? (
              <>
                {/* Cancel (left) */}
                <button onClick={handleClear}
                  className="w-12 h-12 rounded-full bg-gray-200 flex items-center justify-center
                    active:scale-90 transition-transform"
                  aria-label="Cancel">
                  <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>

                {/* Shutter button (center, large) */}
                <button onClick={capturePhoto}
                  disabled={!cameraReady}
                  className="w-[72px] h-[72px] rounded-full bg-white border-[4px] border-leaf-500
                    flex items-center justify-center active:scale-90 transition-transform
                    disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                  aria-label="Capture photo">
                  <div className="w-[56px] h-[56px] rounded-full bg-leaf-500 active:bg-leaf-600 transition-colors" />
                </button>

                {/* Switch camera (right) */}
                {hasMultipleCameras ? (
                  <button onClick={switchCamera}
                    className="w-12 h-12 rounded-full bg-gray-200 flex items-center justify-center
                      active:scale-90 transition-transform"
                    aria-label="Switch camera">
                    <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                  </button>
                ) : (
                  <div className="w-12" /> /* spacer */
                )}
              </>
            ) : (
              <>
                <button onClick={handleClear}
                  className="btn-secondary flex-1 flex items-center justify-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  Cancel
                </button>
                <button onClick={capturePhoto} disabled={!cameraReady}
                  className="btn-primary flex-[2] flex items-center justify-center gap-2 text-lg
                    disabled:opacity-50 disabled:cursor-not-allowed">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    <circle cx="12" cy="13" r="3" strokeWidth={2} />
                  </svg>
                  Capture Photo
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── Preview mode ────────────────────────────── */}
      {mode === "preview" && (
        <div className="space-y-4">
          <div className="relative rounded-2xl overflow-hidden bg-gray-100 border border-gray-200">
            <img src={preview!} alt="Selected leaf"
              className={`w-full object-contain ${isMobile ? "h-[50vh] max-h-[400px]" : "h-64 sm:h-80"}`} />
            <button onClick={handleClear}
              className="absolute top-3 right-3 w-10 h-10 bg-white/90 backdrop-blur-sm rounded-full
                flex items-center justify-center shadow-md hover:bg-white active:scale-90 transition-all"
              aria-label="Remove image">
              <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {errorMessage && (
            <div className="mt-3 mb-1 bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3">
              <div className="text-sm">{errorMessage}</div>
            </div>
          )}

          {selectedFile && (
            <div className="flex items-center justify-between bg-gray-50 rounded-xl px-4 py-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-leaf-100 rounded-lg flex items-center justify-center">
                  <svg className="w-4 h-4 text-leaf-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900 truncate max-w-[200px]">{selectedFile.name}</p>
                  <p className="text-xs text-gray-500">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                </div>
              </div>
            </div>
          )}

          {/* Simple calculator for treatment adjustments */}
          {selectedFile && (
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="text-xs text-gray-600">Inorganic (chemical)</label>
                <select
                  value={inorganicOption}
                  onChange={(e) => {
                    const v = e.target.value;
                    setInorganicOption(v);
                    if (v) {
                      setOrganicOption("");
                      setOrganicOther("");
                      setOrganicAmount("");
                    }
                  }}
                  className="mt-1 input w-full"
                >
                  <option value="">-- select --</option>
                  <option value="mancozeb">Mancozeb</option>
                  <option value="chlorothalonil">Chlorothalonil</option>
                  <option value="azoxystrobin">Azoxystrobin</option>
                  <option value="triazole">Triazole</option>
                  <option value="copper">Copper hydroxide</option>
                  <option value="other">Other (specify)</option>
                </select>
                {inorganicOption === "other" && (
                  <input
                    type="text"
                    value={inorganicOther}
                    onChange={(e) => setInorganicOther(e.target.value)}
                    className="mt-2 input w-full"
                    placeholder="Specify other chemical"
                  />
                )}
              </div>

              <div>
                <label className="text-xs text-gray-600">Organic</label>
                <select
                  value={organicOption}
                  onChange={(e) => {
                    const v = e.target.value;
                    setOrganicOption(v);
                    if (v) {
                      setInorganicOption("");
                      setInorganicOther("");
                      setChemicalAmount("");
                    }
                  }}
                  className="mt-1 input w-full"
                >
                  <option value="">-- select --</option>
                  <option value="neem_oil">Neem oil</option>
                  <option value="bacillus_subtilis">Bacillus subtilis</option>
                  <option value="trichoderma">Trichoderma spp.</option>
                  <option value="compost_tea">Compost tea</option>
                  <option value="other">Other (specify)</option>
                </select>
                {organicOption === "other" && (
                  <input
                    type="text"
                    value={organicOther}
                    onChange={(e) => setOrganicOther(e.target.value)}
                    className="mt-2 input w-full"
                    placeholder="Specify other organic"
                  />
                )}
              </div>

              <div>
                <label className="text-xs text-gray-600">Chemical (g)</label>
                <input
                  type="number"
                  step="any"
                  value={chemicalAmount}
                  onChange={(e) => setChemicalAmount(e.target.value)}
                  disabled={!!organicOption}
                  className="mt-1 input w-full"
                  placeholder="e.g. 50"
                />
              </div>

              <div className="sm:col-span-3">
                <label className="text-xs text-gray-600">Water (L)</label>
                <input
                  type="number"
                  step="any"
                  value={waterAmount}
                  onChange={(e) => setWaterAmount(e.target.value)}
                  className="mt-1 input w-full"
                  placeholder="e.g. 10"
                />
              </div>

              <div>
                <label className="text-xs text-gray-600">Organic (g)</label>
                <input
                  type="number"
                  step="any"
                  value={organicAmount}
                  onChange={(e) => setOrganicAmount(e.target.value)}
                  disabled={!!inorganicOption}
                  className="mt-1 input w-full"
                  placeholder="optional"
                />
              </div>

              <div className="sm:col-span-3">
                <label className="text-xs text-gray-600">Notes (optional)</label>
                <input type="text" value={notes} onChange={(e) => setNotes(e.target.value)} className="mt-1 input w-full" placeholder="e.g. target concentration 0.5%" />
              </div>
            </div>
          )}

          <button onClick={handleAnalyze}
            className={`btn-primary w-full flex items-center justify-center gap-2
              ${isMobile ? "text-lg py-4 rounded-2xl" : "text-lg"}`}>
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            Analyze Disease
          </button>
        </div>
      )}
    </div>
  );
}
