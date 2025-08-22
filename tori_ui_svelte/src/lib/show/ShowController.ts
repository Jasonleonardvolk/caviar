// ShowController.ts - Extended show controller with WOW Pack video clip support

import { loadManifest, findClip } from '$lib/video/wowManifest';
import { chooseBest, getCodecLabel } from '$lib/video/chooseSource';

export class ShowController {
  private videoElement: HTMLVideoElement | null = null;
  private currentClipId: string | null = null;
  
  constructor() {
    // Initialize show controller
  }
  
  /**
   * Try to play a video clip from URL query parameters
   * Returns true if a clip was found and started playing
   */
  async tryPlayClipFromQuery(videoEl: HTMLVideoElement): Promise<boolean> {
    const params = new URLSearchParams(location.search);
    const clipId = params.get('clip');
    
    if (!clipId) {
      console.log('No clip parameter in URL');
      return false;
    }
    
    try {
      // Find the clip in manifest
      const clip = await findClip(clipId);
      if (!clip) {
        console.warn(`Clip not found: ${clipId}`);
        return false;
      }
      
      // Choose best source based on device capabilities
      const bestSource = await chooseBest(clip.sources);
      if (!bestSource) {
        console.error('No compatible source found for clip');
        return false;
      }
      
      console.log(`Playing clip: ${clip.label} (${getCodecLabel(bestSource.type)})`);
      
      // Clear any existing sources
      videoEl.innerHTML = '';
      
      // Create and append the source element
      const sourceEl = document.createElement('source');
      sourceEl.src = bestSource.url;
      sourceEl.type = `video/mp4; codecs="${bestSource.codecs}"`;
      videoEl.appendChild(sourceEl);
      
      // Store references
      this.videoElement = videoEl;
      this.currentClipId = clipId;
      
      // Set video attributes for better playback
      videoEl.loop = true;
      videoEl.muted = true; // Required for autoplay
      videoEl.playsInline = true; // Important for mobile
      videoEl.setAttribute('x-webkit-airplay', 'allow');
      
      // Try to play
      try {
        await videoEl.play();
        console.log('Video playback started successfully');
        return true;
      } catch (playError) {
        console.warn('Autoplay failed, user interaction may be required:', playError);
        
        // Set up click-to-play fallback
        const playOnClick = async () => {
          try {
            await videoEl.play();
            videoEl.removeEventListener('click', playOnClick);
          } catch (e) {
            console.error('Play failed even with user interaction:', e);
          }
        };
        videoEl.addEventListener('click', playOnClick);
        
        return true; // Still return true since clip was loaded
      }
    } catch (error) {
      console.error('Error loading clip:', error);
      return false;
    }
  }
  
  /**
   * Switch to a different clip
   */
  async switchToClip(clipId: string): Promise<boolean> {
    if (!this.videoElement) {
      console.error('No video element set');
      return false;
    }
    
    // Create a new URL with the clip parameter
    const url = new URL(window.location.href);
    url.searchParams.set('clip', clipId);
    window.history.pushState({}, '', url);
    
    // Load the new clip
    return this.tryPlayClipFromQuery(this.videoElement);
  }
  
  /**
   * Get list of all available clips
   */
  async getAvailableClips() {
    try {
      const clips = await loadManifest();
      return clips.map(c => ({
        id: c.id,
        label: c.label,
        sources: c.sources.length
      }));
    } catch (error) {
      console.error('Failed to load clips:', error);
      return [];
    }
  }
  
  /**
   * Get current clip ID
   */
  getCurrentClipId(): string | null {
    return this.currentClipId;
  }
  
  /**
   * Clear video and return to shader mode
   */
  clearVideo() {
    if (this.videoElement) {
      this.videoElement.pause();
      this.videoElement.innerHTML = '';
      this.videoElement = null;
    }
    this.currentClipId = null;
    
    // Remove clip parameter from URL
    const url = new URL(window.location.href);
    url.searchParams.delete('clip');
    window.history.pushState({}, '', url);
  }
}

// Export singleton instance
export const showController = new ShowController();
