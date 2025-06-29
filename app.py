from concurrent.futures import ThreadPoolExecutor
import os
from flask import Flask, render_template, request, jsonify
import math
import requests
import json
from flask_cors import CORS 
import google.generativeai as genai
from openai import OpenAI
from erlang import erlangb


app = Flask(__name__)
CORS(app)

# API keys
OPENROUTER_API_KEY = "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"
DEEPSEEK_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyCl6gI7KYssoJK7P_osPd7W7QAizNHrBlQ"


##############
def call_openrouter(prompt):
    """Call OpenAI via OpenRouter"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=10
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenRouter error: {str(e)}"

##############
import requests


client = OpenAI(api_key="sk-c3141d8760ef47789efabf7ef485a86d", base_url="https://api.deepseek.com")

def call_deepseek(prompt):
    """Call DeepSeek Chat API """
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=10
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenRouter error: {str(e)}"


##############
genai.configure(api_key= GEMINI_API_KEY)  # or use an env variable

def call_gemini(prompt):
    """Call Google Gemini API (gemini-2.5-flash model)"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {str(e)}"


##############
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/calculate', methods=['POST'])
def calculate():
    scenario = request.json.get("scenario")
    inputs = request.json.get("inputs")

    if scenario == "bitrate_pipeline":
        # Extract inputs
        bandwidth = float(inputs.get("bandwidth"))
        quantizer_bits = int(inputs.get("quantizer_bits"))
        source_encoder_rate = float(inputs.get("source_encoder_rate"))
        channel_encoder_rate = float(inputs.get("channel_encoder_rate"))
        interleaver_bits = float(inputs.get("interleaver_bits"))

        # Calculations
        sampling_frequency = 2 * bandwidth
        quantizer_output_bitrate = sampling_frequency * quantizer_bits
        source_encoder_bitrate = quantizer_output_bitrate * source_encoder_rate
        channel_encoder_bitrate = source_encoder_bitrate / channel_encoder_rate
        interleaver_output_bitrate = channel_encoder_bitrate 

        results = {
            "Sampling frequency (samples/sec)": f"{sampling_frequency:.2f}",
            "Quantizer Output Bitrate (bps)": f"{quantizer_output_bitrate:.2f}",
            "Source Encoder Bitrate (bps)": f"{source_encoder_bitrate:.2f}",
            "Channel Encoder Bitrate (bps)": f"{channel_encoder_bitrate:.2f}",
            "Interleaver Output Bitrate (bps)": f"{interleaver_output_bitrate:.2f}",
        }
        prompt = f"""
You are an AI assistant integrated into a wireless network design web application. 
Your role is to clearly explain results from system computations performed by the backend.

The following bitrates were calculated for a wireless communication system:

- Sampling frequency: {sampling_frequency}
- Quantizer Output Bitrate: {quantizer_output_bitrate}
- Source Encoder Bitrate: {source_encoder_bitrate}
- Channel Encoder Bitrate: {channel_encoder_bitrate}
- Interleaver Output Bitrate: {interleaver_output_bitrate}

Explain each stage in simple terms:
- What does this stage do in the communication chain?
- Why does the bitrate change here?
- How do these transformations impact data transmission?

Use everyday analogies and keep your explanation beginner-friendly, suitable for a engineering student.
"""


##############################################################################



    elif scenario == "ofdm":
      total_bandwidth = float(inputs.get("total_bandwidth"))              # Total system bandwidth in Hz
      subcarrier_spacing = float(inputs.get("subcarrier_spacing"))        # Spacing between subcarriers in Hz (e.g., 15 kHz in LTE)
      num_resource_blocks = int(inputs.get("num_resource_blocks"))        # Number of resource blocks allocated
      subcarriers_per_rb = int(inputs.get("subcarriers_per_rb"))          # Number of subcarriers in one resource block
      modulation_order = int(inputs.get("modulation_order"))              # Modulation order (e.g., 2 for QPSK, 4 for 16-QAM, etc.)
      coding_rate = float(inputs.get("coding_rate"))                      # Channel coding rate (e.g., 0.5 for 1/2 rate coding)

        # Overhead caused by Cyclic Prefix (CP) — about 7% longer symbol duration in typical LTE systems
      cyclic_prefix_overhead = 0.07
        # Total number of subcarriers used in the transmission
      total_subcarriers = num_resource_blocks * subcarriers_per_rb
        # Bits per modulation symbol = log2(M), where M is modulation order
      m = math.log2(modulation_order)
        # Bits per Resource Element (RE) after coding = bits per symbol × coding rate
      bits_per_re = m * coding_rate

        # Useful symbol duration (Tu), without cyclic prefix
      t_u = 1.0 / subcarrier_spacing
        # Total OFDM symbol duration including CP overhead
      t_sym = t_u * (1 + cyclic_prefix_overhead)
        # Number of OFDM symbols transmitted per second on a single subcarrier
      symbols_per_second = 1.0 / t_sym

        # Data rate per subcarrier (RE) = bits per RE × number of symbols per second
      rate_per_re = bits_per_re * symbols_per_second
        # Data rate per Resource Block (RB) = number of subcarriers per RB × rate per RE
      rate_per_rb = subcarriers_per_rb * rate_per_re
        # Maximum total capacity = rate per RB × number of RBs
      max_capacity = num_resource_blocks * rate_per_rb
        # Spectral Efficiency = Total capacity / Bandwidth → in bits/sec/Hz
      spectral_efficiency = max_capacity / total_bandwidth

      results = {
          "Data rate per Resource Element (bps)": f"{rate_per_re:.2f}",
          "Data rate per OFDM symbol (per subcarrier) (bps)": f"{rate_per_re:.2f}",
          "Data rate per Resource Block (bps)": f"{rate_per_rb:.2f}",
          "Maximum transmission capacity (bps)": f"{max_capacity:.2f}",
          "Spectral efficiency (bps/Hz)": f"{spectral_efficiency:.4f}"
      }


      prompt = f"""
You are an AI assistant embedded in a web app for wireless and OFDM system design. 
The backend has computed various performance metrics based on user input.

Given:
- Total bandwidth: {total_bandwidth} Hz
- Subcarrier spacing: {subcarrier_spacing} Hz
- Number of resource blocks: {num_resource_blocks}
- Subcarriers per resource block: {subcarriers_per_rb}
- Modulation order: {modulation_order}
- Coding rate: {coding_rate}
- Cyclic prefix overhead: 7%

Computed results:
- Data rate per Resource Element: {rate_per_re:.2f} bps
- Data rate per OFDM symbol (per subcarrier): {rate_per_re:.2f} bps
- Data rate per Resource Block: {rate_per_rb:.2f} bps
- Maximum transmission capacity: {max_capacity:.2f} bps
- Spectral efficiency: {spectral_efficiency:.4f} bps/Hz

Please explain:
- What each parameter means and how it affects the OFDM system
- How the computed values are derived and what they represent
- Why these metrics are important in evaluating wireless performance

Use easy-to-understand language and analogies to help students and engineers quickly grasp the concepts.
"""

##############################################################################



    elif scenario == "link_budget":
        freq_mhz = float(inputs.get("freq_mhz"))                  # Frequency in MHz
        distance_km = float(inputs.get("distance_km"))            # Distance between Tx and Rx in km
        tx_power_dbm = float(inputs.get("tx_power_dbm"))          # Transmit power in dBm
        tx_gain_dbi = float(inputs.get("tx_gain_dbi"))            # Transmitter antenna gain in dBi
        tx_cable_loss_db = float(inputs.get("tx_cable_loss_db"))  # Loss in the Tx cable in dB
        rx_gain_dbi = float(inputs.get("rx_gain_dbi"))            # Receiver antenna gain in dBi
        rx_cable_loss_db = float(inputs.get("rx_cable_loss_db"))  # Loss in the Rx cable in dB

        # Calculate Free Space Path Loss (FSPL) in dB using the standard formula:
        # FSPL(dB) = 20*log10(distance_km) + 20*log10(freq_MHz) + 32.44
        fspl = 20 * math.log10(distance_km) + 20 * math.log10(freq_mhz) + 32.44

        # Effective transmitted power after accounting for transmitter cable loss
        effective_tx_power = tx_power_dbm - tx_cable_loss_db
        
        # Received power (in dBm) = Effective Tx power + Tx gain + Rx gain - Rx cable loss - FSPL
        received_power_dbm = (tx_power_dbm - tx_cable_loss_db) + tx_gain_dbi + rx_gain_dbi - rx_cable_loss_db - fspl

        results = {
            "Effective Transmitted Power (dBm)": f"{effective_tx_power:.2f}",
            "Free Space Path Loss (dB)": f"{fspl:.2f}",
            "Received Signal Strength (dBm)": f"{received_power_dbm:.2f}",
        }

        prompt = f"""
You are an AI agent supporting a wireless design tool. The backend has already performed a link budget analysis using these user inputs:

- Frequency: {freq_mhz} MHz
- Distance: {distance_km} km
- Transmitter Power: {tx_power_dbm} dBm
- Transmitter Antenna Gain: {tx_gain_dbi} dBi
- Transmitter Cable Loss: {tx_cable_loss_db} dB
- Receiver Antenna Gain: {rx_gain_dbi} dBi
- Receiver Cable Loss: {rx_cable_loss_db} dB

Calculated values:
- Effective Transmitted Power: {effective_tx_power:.2f} dBm
- Free Space Path Loss (FSPL): {fspl:.2f} dB
- Received Signal Strength: {received_power_dbm:.2f} dBm

Provide a clear explanation for:
- What each parameter and result means
- How they affect signal strength and communication range
- Why these results are critical for designing reliable wireless links

Keep your explanation simple, step-by-step, and educational for junior engineers or students.
"""



##############################################################################



    elif scenario == "cellular_design":
        total_area = float(inputs.get("total_area"))                 # Total coverage area in km²
        cell_radius = float(inputs.get("cell_radius"))               # Radius of each hexagonal cell 
        cluster_size = int(inputs.get("cluster_size"))               # Frequency reuse cluster size (e.g., 3, 7)
        total_bandwidth = float(inputs.get("total_bandwidth"))       # Total system bandwidth kHz
        channel_bandwidth = float(inputs.get("channel_bandwidth"))   # Bandwidth per channel  kHz
        num_users = int(inputs.get("num_users"))                     # Total number of users in the system
        traffic_per_user = float(inputs.get("traffic_per_user"))     # Offered traffic per user in Erlangs
        blocking_target = float(inputs.get("blocking_target"))       # Target blocking probability (0.02 for 2%)


        # Calculate cell area 
        cell_area = (3 * math.sqrt(3) / 2) * (cell_radius ** 2)
        # Calculate the number of cells 
        num_cells = math.ceil(total_area / cell_area)
        # Calculate total number of channels in the system
        total_channels = math.floor(total_bandwidth * 1000 / channel_bandwidth)
        # Channels allocated per cell
        channels_per_cell = total_channels // cluster_size
        # Total offered traffic in Erlangs for all users
        total_traffic = num_users * traffic_per_user
        # Average traffic per cell
        traffic_per_cell = total_traffic / num_cells if num_cells > 0 else 0

        low, high = 0.0, 1000.0 * channels_per_cell
        max_traffic_per_cell = 0.0

        for _ in range(100):
            mid = (low + high) / 2
            b_mid = erlangb(traffic=mid, channels=channels_per_cell)
            if b_mid < blocking_target:
                low = mid
            else:
                high = mid
            if abs(high - low) < 1e-4:
                break
        max_traffic_per_cell = low


        # Compute maximum users per cell 
        max_users_per_cell = max_traffic_per_cell / traffic_per_user if traffic_per_user > 0 else float('inf')
        #Compute maximum users per system
        max_system_users = max_users_per_cell * num_cells

        # Compute current blocking probability
        blocking_prob = erlangb(traffic=traffic_per_cell, channels=channels_per_cell) if channels_per_cell > 0 else 1.0

        results = {
            "Hexagonal cell area (km²)": f"{cell_area:.2f}",
            "Number of cells required": num_cells,
            "Total available channels": total_channels,
            "Channels per cell": channels_per_cell,
            "Total offered traffic (Erlangs)": f"{total_traffic:.2f}",
            "Traffic per cell (Erlangs)": f"{traffic_per_cell:.2f}",
            "Computed blocking probability": f"{blocking_prob:.4f}",
            "Target blocking probability": blocking_target,
            "Maximum traffic per cell (Erlangs)": f"{max_traffic_per_cell:.2f}",
            "Maximum users per cell": f"{max_users_per_cell:.2f}",
            "Maximum users in system": f"{max_system_users:.2f}"
        }

        prompt = f"""
You are a built-in AI assistant in a cellular network design web platform. 
The backend has used user-provided inputs to calculate several planning metrics.

Calculated outputs:
- Hexagonal cell area: {cell_area:.2f} km²
- Required number of cells: {num_cells}
- Total available channels: {total_channels}
- Channels per cell (reuse factor applied): {channels_per_cell}
- Total system traffic: {total_traffic:.2f} Erlangs
- Traffic per cell: {traffic_per_cell:.2f} Erlangs
- Target blocking probability: {blocking_target}
- Maximum traffic per cell (meeting blocking target): {max_traffic_per_cell:.2f} Erlangs
- Maximum users per cell: {max_users_per_cell:.2f}
- Maximum users system-wide: {max_system_users:.2f}
- Computed blocking probability: {blocking_prob:.4f}

Key user inputs:
- Total coverage area: {total_area} km²
- Cell radius: {cell_radius} km
- Frequency reuse factor (K): {cluster_size}
- Total bandwidth: {total_bandwidth} MHz
- Channel bandwidth: {channel_bandwidth} kHz
- Number of users: {num_users}
- Average traffic per user: {traffic_per_user} Erlangs
- Blocking probability target: {blocking_target}

Now, explain:
- What each result means in the context of cellular system planning
- Why it's important (e.g., ensuring service quality, managing interference)
- How user inputs affect these outputs

Use simple terms, real-world analogies (e.g., “like traffic lanes on a highway”), and ensure clarity for students and junior engineers.
"""


    else:
        return jsonify({"error": "Unknown scenario"}), 400

    ai_responses = {}
    with ThreadPoolExecutor() as executor:
        # Submit all API calls
        future_openrouter = executor.submit(call_openrouter, prompt)
        future_deepseek = executor.submit(call_deepseek, prompt)
        future_gemini = executor.submit(call_gemini, prompt)
        
        # Get results
        ai_responses = {
            "DeepSeek V3": future_deepseek.result(),
            "Gemini 2.5-flash": future_gemini.result(),
            "OepnAI GPT-4o-mini": future_openrouter.result(),
        }

    return jsonify({
        "results": results,
        "explanations": ai_responses
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)



