/**
 * Prithvi Quant Labs - Main JavaScript
 * Live NIFTY 50 Ticker & Interactive Features
 */

// Store current stock data for in-place updates
let currentStocksData = [];

// Save animation position before updates
let tickerAnimationPosition = 0;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize live ticker
    initLiveTicker();
    
    // Set up auto-refresh every 30 seconds (but don't reset animation)
    setInterval(updateTickerData, 30000);
});

/**
 * Initialize Live NIFTY 50 Ticker
 */
async function initLiveTicker() {
    const tickerContainer = document.getElementById('ticker-content');
    if (!tickerContainer) return;
    
    // Show loading state
    tickerContainer.innerHTML = '<div class="ticker-loading">Loading market data...</div>';
    
    try {
        const response = await fetch('/api/live-ticker/');
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        
        if (data.success && data.stocks && data.stocks.length > 0) {
            currentStocksData = data.stocks;
            renderTicker(data.stocks);
        } else {
            // Fallback to static data if API fails
            renderStaticTicker();
        }
    } catch (error) {
        console.error('Error fetching ticker data:', error);
        // Show static fallback data
        renderStaticTicker();
    }
}

/**
 * Update ticker data without resetting animation
 * This refreshes prices/changes while keeping the animation seamless
 */
async function updateTickerData() {
    try {
        const response = await fetch('/api/live-ticker/');
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        
        if (data.success && data.stocks && data.stocks.length > 0) {
            currentStocksData = data.stocks;
            updateTickerValues(data.stocks);
        }
        // If API fails, keep using existing data (no update needed)
    } catch (error) {
        console.error('Error updating ticker data:', error);
        // Silently fail - keep showing existing data
    }
}

/**
 * Update ticker values in-place without replacing HTML (preserves animation)
 */
function updateTickerValues(stocks) {
    const tickerContainer = document.getElementById('ticker-content');
    if (!tickerContainer) return;
    
    // Get all ticker items (both sets for seamless loop)
    const allItems = tickerContainer.querySelectorAll('.ticker-item');
    
    stocks.forEach((stock, index) => {
        if (allItems[index]) {
            const priceEl = allItems[index].querySelector('.ticker-price');
            const changeEl = allItems[index].querySelector('.ticker-change');
            
            if (priceEl) {
                priceEl.textContent = formatPrice(stock.price);
            }
            
            if (changeEl) {
                const isUp = stock.change >= 0;
                const changeClass = isUp ? 'up' : 'down';
                const arrow = isUp ? '▲' : '▼';
                const changeSign = isUp ? '+' : '';
                const changeFormatted = typeof stock.change === 'number' 
                    ? `${changeSign}${stock.change.toFixed(2)}%` 
                    : '0.00%';
                
                changeEl.className = `ticker-change ${changeClass}`;
                changeEl.innerHTML = `${arrow} ${changeFormatted}`;
            }
        }
    });
}

/**
 * Render ticker with live data - creates seamless loop
 */
function renderTicker(stocks) {
    const tickerContainer = document.getElementById('ticker-content');
    if (!tickerContainer) return;
    
    // Create ticker items wrapped in ticker-track for animation
    let tickerHTML = '<div class="ticker-track">';
    
    // First set
    stocks.forEach(stock => {
        tickerHTML += createTickerItem(stock);
    });
    
    // Duplicate for seamless scroll (second set)
    stocks.forEach(stock => {
        tickerHTML += createTickerItem(stock);
    });
    
    tickerHTML += '</div>';
    
    tickerContainer.innerHTML = tickerHTML;
}

/**
 * Create single ticker item HTML
 */
function createTickerItem(stock) {
    const isUp = stock.change >= 0;
    const changeClass = isUp ? 'up' : 'down';
    const arrow = isUp ? '▲' : '▼';
    const changeSign = isUp ? '+' : '';
    const changeFormatted = typeof stock.change === 'number' 
        ? `${changeSign}${stock.change.toFixed(2)}%` 
        : '0.00%';
    
    return `
        <div class="ticker-item">
            <span style="color:#a5b4fc;font-weight:600;font-size:20px;margin-right:8px;">${stock.symbol}</span>
            <span class="ticker-price" style="color:white;font-weight:700;font-size:20px;">${formatPrice(stock.price)}</span>
            <span class="ticker-change ${changeClass}" style="font-size:18px;margin-left:8px;">
                ${arrow} ${changeFormatted}
            </span>
        </div>
    `;
}

/**
 * Render static fallback ticker
 */
function renderStaticTicker() {
    // Static NIFTY 50 stocks as fallback
    const staticStocks = [
        { symbol: 'NIFTY50', price: 22450.00, change: 0.45 },
        { symbol: 'BANKNIFTY', price: 48500.00, change: 0.32 },
        { symbol: 'RELIANCE', price: 2985.20, change: 1.25 },
        { symbol: 'TCS', price: 4200.50, change: -0.45 },
        { symbol: 'HDFCBANK', price: 1680.30, change: 0.85 },
        { symbol: 'INFY', price: 1520.75, change: -0.32 },
        { symbol: 'ICICIBANK', price: 1185.40, change: 0.65 },
        { symbol: 'KOTAKBANK', price: 1850.20, change: 1.10 },
        { symbol: 'SBIN', price: 780.50, change: -0.55 },
        { symbol: 'LT', price: 3650.80, change: 0.95 },
        { symbol: 'ITC', price: 425.30, change: 0.25 },
        { symbol: 'HINDUNILVR', price: 2850.60, change: -0.15 },
        { symbol: 'ASIANPAINT', price: 3150.40, change: 0.75 },
        { symbol: 'BAJFINANCE', price: 7250.90, change: 1.45 },
        { symbol: 'MARUTI', price: 12850.00, change: -0.85 }
    ];
    
    currentStocksData = staticStocks;
    renderTicker(staticStocks);
}

/**
 * Format price with Indian numbering system
 */
function formatPrice(price) {
    if (price === null || price === undefined || isNaN(price)) {
        return '--';
    }
    
    // Format with commas (Indian system)
    if (price >= 10000000) { // 1 crore
        return '₹' + (price / 10000000).toFixed(2) + 'Cr';
    } else if (price >= 100000) { // 1 lakh
        return '₹' + (price / 100000).toFixed(2) + 'L';
    } else {
        return '₹' + price.toLocaleString('en-IN', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    }
}

/**
 * Smooth scroll to section
 */
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Add smooth reveal animation on scroll
window.addEventListener('scroll', function() {
    const cards = document.querySelectorAll('.premium-card');
    
    cards.forEach((card, index) => {
        const rect = card.getBoundingClientRect();
        const isVisible = rect.top < window.innerHeight - 100;
        
        if (isVisible) {
            card.classList.add('fade-in');
            card.classList.add(`stagger-${index + 1}`);
        }
    });
});

// Initialize animations on page load
document.querySelectorAll('.premium-card').forEach((card, index) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    
    setTimeout(() => {
        card.style.opacity = '1';
        card.style.transform = 'translateY(0)';
    }, 100 * (index + 1));
});

