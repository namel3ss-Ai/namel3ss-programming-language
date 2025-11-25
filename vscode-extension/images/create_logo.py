#!/usr/bin/env python3
"""
Simple script to create a namel3ss logo PNG
"""

def create_simple_logo_data():
    """Create a simple logo as base64 encoded PNG data"""
    # This is a 128x128 white circle with basic text representation
    # We'll create a minimal PNG header and data
    import struct
    import zlib
    
    width, height = 128, 128
    
    # Create RGBA data for a simple white logo
    def make_rgba_data():
        data = []
        center_x, center_y = 64, 64
        radius = 60
        
        for y in range(height):
            row = []
            for x in range(width):
                # Check if we're in the circle
                dx, dy = x - center_x, y - center_y
                if dx*dx + dy*dy <= radius*radius:
                    # Inside circle - white background
                    row.extend([255, 255, 255, 255])  # White
                else:
                    # Outside circle - transparent
                    row.extend([0, 0, 0, 0])  # Transparent
            data.extend(row)
        
        # Add simple text representation (approximate)
        # "N3" around y=45-60, "ai" around y=75-85
        for y in range(45, 65):  # N3 area
            for x in range(45, 85):  # N3 area
                if (x-50 < 8 or x-75 > -8) and y < 60:  # Rough N3 shape
                    idx = (y * width + x) * 4
                    if idx < len(data) - 3:
                        data[idx:idx+3] = [31, 41, 55]  # Dark gray for N3
        
        for y in range(75, 90):  # ai area  
            for x in range(55, 75):  # ai area
                if (x-58 < 4 or x-68 < 4) and y < 85:  # Rough ai shape
                    idx = (y * width + x) * 4
                    if idx < len(data) - 3:
                        data[idx:idx+3] = [0, 0, 0]  # Black for ai
        
        return bytes(data)
    
    rgba_data = make_rgba_data()
    
    # Create PNG data
    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return struct.pack("!I", len(data)) + chunk_head + struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head))
    
    png_bytes = b'\\x89PNG\\r\\n\\x1a\\n'
    png_bytes += png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0))
    
    # Compress the RGBA data
    compressor = zlib.compressobj()
    png_data = b''
    for y in range(height):
        png_data += b'\\x00'  # Filter type 0 (None)
        png_data += rgba_data[y * width * 4:(y + 1) * width * 4]
    
    compressed = compressor.compress(png_data)
    compressed += compressor.flush()
    
    png_bytes += png_pack(b'IDAT', compressed)
    png_bytes += png_pack(b'IEND', b'')
    
    return png_bytes

if __name__ == "__main__":
    logo_data = create_simple_logo_data()
    
    with open('logo.png', 'wb') as f:
        f.write(logo_data)
    
    print("✅ Created logo.png")
    print("📏 Size: 128x128 pixels")
    print("🎨 Design: White background, 'N3' in dark gray, 'ai' in black")