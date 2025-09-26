let lights = document.querySelectorAll(".light");
let currentLight = 0;
let greenDuration = 3000; // default (ms)

// Fetch latest green duration from Flask API
async function updateGreenTime() {
    try {
        let response = await fetch("http://127.0.0.1:5000/get_green_time");
        if (!response.ok) throw new Error("Network error");
        
        let data = await response.json();
        if (data && data.duration) {
            greenDuration = data.duration * 1000; // sec -> ms
        }
    } catch (err) {
        console.warn("⚠️ Using fallback green time. Error:", err.message);
    }
}

// Update greenDuration every 2s
setInterval(updateGreenTime, 2000);

// Initialize first light
lights[currentLight].classList.add("active");

function changeLight() {
    // Remove active class from current light
    lights[currentLight].classList.remove("active");

    // Move to next light
    currentLight = (currentLight + 1) % lights.length;

    // Add active class to new light
    lights[currentLight].classList.add("active");

    // Set timing: green → dynamic, others → fixed
    let delay = (currentLight === 2) ? greenDuration : 2000;

    // Schedule next change
    setTimeout(changeLight, delay);
}

// Start the cycle after initial green
setTimeout(changeLight, greenDuration);
