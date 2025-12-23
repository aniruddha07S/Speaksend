import cv2
import time

def test_camera():
    print("\n🔍 Testing camera access...")
    
    # Try different camera backends
    backends = [
        (cv2.CAP_AVFOUNDATION, "AVFoundation"),
        (cv2.CAP_ANY, "Default"),
        (cv2.CAP_DSHOW, "DirectShow")
    ]
    
    for backend, name in backends:
        print(f"\nTrying {name} backend...")
        try:
            # Initialize camera with specific backend
            cap = cv2.VideoCapture(0, backend)
            
            if not cap.isOpened():
                print(f"❌ Failed to open camera with {name} backend")
                continue
                
            print(f"✅ Camera opened successfully with {name} backend")
            
            # Try to read a frame
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Failed to read frame with {name} backend")
            else:
                print(f"✅ Successfully read frame with {name} backend")
                print(f"Frame shape: {frame.shape}")
                
                # Create window
                window_name = f'Camera Test - {name}'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)
                
                print("\n📸 Camera test running...")
                print("Press 'q' to quit the camera test")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Failed to read frame")
                        break
                    
                    # Display the frame
                    cv2.imshow(window_name, frame)
                    
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("👋 Camera test stopped by user")
                        break
                
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"❌ Error with {name} backend: {str(e)}")
    
    print("\n📝 Camera Test Summary:")
    print("1. If no backends worked, please check:")
    print("   - Camera permissions in System Preferences > Security & Privacy > Privacy > Camera")
    print("   - No other application is using the camera")
    print("   - Camera is properly connected")
    print("2. If some backends worked, we'll use the first working one")

if __name__ == "__main__":
    test_camera() 