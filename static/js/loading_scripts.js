$(document).ready(function() {

    if (screen.width / screen.height >  3 / 4) {
      // GSAP ScrollTrigger
      gsap.registerPlugin(ScrollTrigger);
  
      let tl;
  
      // Create a GSAP timeline
      tl = gsap.timeline({
        paused: true, // Pause the timeline initially
        reversed: true, // Begin in the reversed state (animation end)
      });
  
      gsap.to(tl, {progress: 1, duration: 5, ease: "power1.inOut"}, 0);
      gsap.to(tl, {
        progress: 0, 
        duration: 1,
        ease: "power1.inOut", 
      }, 0.2);
      gsap.to('#loading-screen', {
        opacity: 0,
        zIndex: -99,
        duration: 0.5,
        ease: "power1.inOut", 
        onComplete: function() {
          //const loadScreen = document.getElementById('loading-screen');
          loadScreen.style.display = 'none';
        }
      }, 0.2);
  
      gsap.to('.header-top', {
        opacity: 1,
        duration: 1,
        ease: "power1.inOut", 
        onComplete: function() {
          //const loadScreen = document.getElementById('loading-screen');
          start_mouse(tl);
        }
      }, 0.2);
      
    }
  });